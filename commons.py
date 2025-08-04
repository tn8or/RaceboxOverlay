import asyncio
import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from multiprocessing import cpu_count

import ffmpeg
from PIL import Image, ImageDraw, ImageFont


def setup_logging():
    logger = logging.getLogger()
    logger.handlers.clear()

    output = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
    )
    output.setFormatter(formatter)

    logger.addHandler(output)
    logger.setLevel(logging.INFO)

    return logger


logger = setup_logging()


class dashGenerator:
    def __init__(self, rows, header, width, filename):

        self.tmpdir = tempfile.TemporaryDirectory(dir="/tmp")
        logger.info("Will work in %s", self.tmpdir.name)

        # initialize global vars
        self.maxspeed = float(0)
        self.maxleanleft = float(0)
        self.maxleanright = float(0)
        self.maxaccg = float(0)
        self.maxdecg = float(0)
        self.maxlatitude = float(0)
        self.minlatitude = float(360)
        self.maxlongitude = float(0)
        self.minlongitude = float(360)
        self.filename = filename
        self.rows = rows
        self.header = header
        self.foldername = self.tmpdir.name + "/"
        self.log = []
        self.laps = []
        self.hasShownLaps = False
        self.prevBest = 9999999999
        self.trackimage = None

        if width == 1920:
            self.fontsize = 50
            self.height = 1080
            self.width = width
            self.maxtracksize = 500
            self.polygonwidth = 8
            self.polygonmargin = 10
            self.positionsize = 8
        elif width == 3840:
            self.fontsize = 100
            self.width = width
            self.height = 2160
            self.maxtracksize = 1000
            self.polygonwidth = 16
            self.polygonmargin = 20
            self.positionsize = 16
        else:
            self.fontsize = 30
            self.height = 720
            self.width = width
            self.maxtracksize = 300
            self.polygonwidth = 5
            self.polygonmargin = 8
            self.positionsize = 5

        # Add this block to set self.font
        try:
            self.font = ImageFont.truetype(
                "font/DejaVuSansCondensed-Bold.ttf", size=self.fontsize
            )
        except Exception as e:
            logger.warning("Could not load font, using default: %s", e)
            self.font = ImageFont.load_default()

        if not os.path.exists(self.foldername):
            os.mkdir(self.foldername)
        # iterate through rows to generate statistics
        for row in rows:
            if float(row["Speed"]) > float(self.maxspeed):
                self.maxspeed = row["Speed"]
            # negative lean is left, maybe
            if (
                float(row["LeanAngle"]) < 0
                and -float(row["LeanAngle"]) > self.maxleanright
            ):
                self.maxleanright = -float(row["LeanAngle"])
            if float(row["LeanAngle"]) > self.maxleanleft:
                self.maxleanleft = float(row["LeanAngle"])
            if float(row["GForceX"]) < 0 and -float(row["GForceX"]) > self.maxdecg:
                self.maxdecg = -float(row["GForceX"])
            if float(row["GForceX"]) > self.maxaccg:
                self.maxaccg = float(row["GForceX"])
            if float(row["Latitude"]) > self.maxlatitude:
                self.maxlatitude = float(row["Latitude"])
            if float(row["Latitude"]) < self.minlatitude:
                self.minlatitude = float(row["Latitude"])
            if float(row["Longitude"]) > self.maxlongitude:
                self.maxlongitude = float(row["Longitude"])
            if float(row["Longitude"]) < self.minlongitude:
                self.minlongitude = float(row["Longitude"])

        logger.info("latitudes %s %s", self.minlatitude, self.minlongitude)

        # G-force sanity check: only show G meter if we have both positive and negative values
        self.show_g_meter = self.maxaccg > 0.1 and self.maxdecg > 0.1
        if not self.show_g_meter:
            logger.info(
                "G meter disabled: insufficient data variation (acc: %.2f, dec: %.2f)",
                self.maxaccg,
                self.maxdecg,
            )
        else:
            logger.info(
                "G meter enabled: acc: %.2f, dec: %.2f", self.maxaccg, self.maxdecg
            )

        self.genTrackPolygon()

        for row in header:
            if row.startswith("Lap "):
                lap = row.split(",")
                if len(lap) > 1:
                    lap = float(lap[1].strip())
                    if lap not in self.laps:
                        self.laps.append(lap)

        logger.info("found %s laps", len(self.laps))
        logger.info("laps: %s", self.laps)
        logger.info("general stats gathered")

        # Initialize speed graph data structures
        self.speed_graph_data = {}  # {lap_number: [(frame_index, speed), ...]}
        self.current_lap_speeds = []  # Current lap speed data
        self.speed_graph_width = int(self.width * 0.2)  # 20% screen width
        self.speed_graph_height = int(self.height * 0.08)  # 8% screen height
        self.max_laps_shown = 8  # Show only 8 laps max
        self._preprocess_speed_data()

        # Initialize arrow_images dictionary
        self.arrow_images = {}
        for angle in range(-90, 91):
            path = f"angles/arrow_{angle:03d}.png"
            if os.path.exists(path):
                self.arrow_images[angle] = Image.open(path).convert("RGBA")

    def _get_start_timecode(self):
        """Extract start timestamp from first CSV row and convert to FFmpeg timecode format"""
        if not self.rows:
            return "00:00:00:00"

        try:
            # Parse the timestamp from the first row: "2025-06-29T11:01:39.000Z"
            first_timestamp = self.rows[0]["Time"]
            # Split by 'T' to separate date and time
            time_part = first_timestamp.split("T")[1]
            # Remove the '.000Z' suffix and split by ':'
            time_clean = time_part.split(".")[0]  # "11:01:39"
            hours, minutes, seconds = time_clean.split(":")

            # FFmpeg timecode format is HH:MM:SS:FF (where FF is frame number)
            # Since we start at frame 0, frames = 00
            timecode = f"{hours}:{minutes}:{seconds}:00"
            logger.info(
                f"Setting video timecode to: {timecode} (from CSV start: {first_timestamp})"
            )
            return timecode
        except Exception as e:
            logger.warning(f"Could not parse start timestamp, using default: {e}")
            return "00:00:00:00"

    def convert_seconds(self, seconds):
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return minutes, remaining_seconds

    def _preprocess_speed_data(self):
        """Preprocess speed data for each lap to enable graph visualization"""
        logger.info("Preprocessing speed data for lap graphs...")

        current_lap = 0
        lap_speeds = []
        lap_start_frame = None

        for i, row in enumerate(self.rows):
            lap_num = int(row["Lap"])
            speed = float(row["Speed"])
            frame_num = int(row["Record"])

            if lap_num != current_lap:
                # Store the previous lap's data
                if current_lap > 0 and lap_speeds:
                    self.speed_graph_data[current_lap] = lap_speeds.copy()

                # Start new lap
                current_lap = lap_num
                lap_speeds = []
                lap_start_frame = frame_num

            if (
                lap_num > 0 and lap_start_frame is not None
            ):  # Only store data for actual laps (not lap 0)
                # Calculate progress as frames from lap start
                progress_frames = frame_num - lap_start_frame
                lap_speeds.append((progress_frames, speed))

        # Store the last lap
        if current_lap > 0 and lap_speeds:
            self.speed_graph_data[current_lap] = lap_speeds.copy()

        logger.info(f"Speed data preprocessed for {len(self.speed_graph_data)} laps")

    def _get_lap_color(self, lap_num, current_lap, laps_to_show):
        """Get color for lap line based on age. Green → Yellow → Red progression"""
        if lap_num == current_lap and current_lap > 0:
            return (0, 255, 0, 255)  # Current lap - bright green

        # Calculate age (how many laps ago this was)
        if current_lap > 0:
            age = current_lap - lap_num
        else:
            # We're in out-lap (lap 0), so calculate age from most recent lap
            max_lap = max(laps_to_show) if laps_to_show else 1
            age = max_lap - lap_num

        # Color progression: Green (0) → Yellow (3) → Red (7+) with decreasing opacity
        if age == 0:
            return (0, 255, 0, 240)  # Recent lap - green, high opacity
        elif age == 1:
            return (128, 255, 0, 200)  # Yellow-green, good opacity
        elif age == 2:
            return (255, 255, 0, 160)  # Yellow, medium opacity
        elif age == 3:
            return (255, 192, 0, 130)  # Orange-yellow, lower opacity
        elif age == 4:
            return (255, 128, 0, 100)  # Orange, low opacity
        elif age == 5:
            return (255, 64, 0, 80)  # Red-orange, very low opacity
        else:
            # Older laps fade to red with very low alpha
            alpha = max(30, 70 - (age - 6) * 10)
            return (255, 0, 0, alpha)

    def _get_current_lap_progress(self, current_frame, lap_num):
        """Get the current position within the active lap as a ratio (0-1)"""
        if lap_num <= 0:
            return 0

        # Find the start and total frames for this lap
        lap_start_frame = None
        lap_total_frames = 0

        for row in self.rows:
            if int(row["Lap"]) == lap_num:
                if lap_start_frame is None:
                    lap_start_frame = int(row["Record"])
                lap_total_frames += 1

        if lap_start_frame is None or lap_total_frames == 0:
            return 0

        # Calculate progress as ratio
        frames_into_lap = current_frame - lap_start_frame
        progress = max(0, min(1, frames_into_lap / lap_total_frames))
        return progress

    def draw_speed_graph(self, draw, current_row):
        """Draw speed graph showing current and previous laps"""
        current_lap = int(current_row["Lap"])
        current_frame = int(current_row["Record"])

        # Only show graph during active laps (> 0) or after completing laps
        # Don't show during lap 0 unless we've completed actual laps
        if current_lap == 0 and not self.hasShownLaps:
            return

        # Only show if we're currently in a lap or have completed laps
        if current_lap == 0 and len(self.speed_graph_data) == 0:
            return

        # Graph position - above the speed indicator in bottom left
        graph_x = int(self.width * 0.08)
        graph_y = int(self.height * 0.8 - self.speed_graph_height - self.fontsize * 0.2)

        # Create semi-transparent background
        draw.rounded_rectangle(
            xy=(
                graph_x - 5,
                graph_y - 5,
                graph_x + self.speed_graph_width + 5,
                graph_y + self.speed_graph_height + 5,
            ),
            radius=8,
            fill=(30, 30, 30, 120),
            outline=(100, 100, 100, 180),
            width=1,
        )

        # Determine which laps to show (max 8, current + 7 previous)
        if current_lap > 0:
            # Active lap: show current + previous laps
            laps_to_show = list(range(max(1, current_lap - 7), current_lap + 1))
        else:
            # Out-lap (lap 0): show the most recent completed laps
            available_laps = sorted(self.speed_graph_data.keys())
            laps_to_show = available_laps[-self.max_laps_shown :]

        if not laps_to_show:
            return

        # Calculate maximum lap length (in frames/progress) for normalization
        # This ensures all laps scale to the same width regardless of how long they took
        max_lap_length = 0
        for lap_num in laps_to_show:
            if lap_num in self.speed_graph_data:
                lap_data = self.speed_graph_data[lap_num]
                if lap_data:
                    # Find the maximum progress value (last frame of the lap)
                    max_progress = max(frame_idx for frame_idx, speed in lap_data)
                    max_lap_length = max(max_lap_length, max_progress)

        if max_lap_length == 0:
            return

        # Draw each lap line
        for i, lap_num in enumerate(laps_to_show):
            if lap_num not in self.speed_graph_data:
                continue

            lap_data = self.speed_graph_data[lap_num]
            if not lap_data:
                continue

            # Get color based on lap age (green->yellow->red progression)
            color = self._get_lap_color(lap_num, current_lap, laps_to_show)

            # Set line width
            if lap_num == current_lap and current_lap > 0:
                line_width = 3
            else:
                line_width = 2

            # For progressive drawing, determine how much of this lap to show
            if lap_num == current_lap and current_lap > 0:
                # Current lap: show progress up to current frame
                progress = self._get_current_lap_progress(current_frame, lap_num)
                points_to_show = int(len(lap_data) * progress)
                lap_data_to_draw = (
                    lap_data[:points_to_show] if points_to_show > 0 else []
                )
            elif lap_num < current_lap or (current_lap == 0 and lap_num > 0):
                # Completed laps: show entire lap
                lap_data_to_draw = lap_data
            else:
                # Future laps: don't show
                lap_data_to_draw = []

            # Convert lap data to screen coordinates
            points = []
            for frame_idx, speed in lap_data_to_draw:
                # Normalize frame position by maximum lap length (distance-based scaling)
                x_pos = graph_x + (frame_idx / max_lap_length) * self.speed_graph_width
                # Normalize speed (0 to maxspeed)
                y_pos = (
                    graph_y
                    + self.speed_graph_height
                    - (speed / float(self.maxspeed)) * self.speed_graph_height
                )
                points.append((x_pos, y_pos))

            # Draw the line if we have enough points
            if len(points) >= 2:
                # Draw line segments directly (PIL handles alpha automatically in RGBA mode)
                for j in range(len(points) - 1):
                    draw.line([points[j], points[j + 1]], fill=color, width=line_width)

                # If this is the current lap, show progress indicator
                if lap_num == current_lap and current_lap > 0 and len(points) > 0:
                    # Highlight current position (last point drawn)
                    current_pos = points[-1]
                    draw.ellipse(
                        (
                            current_pos[0] - 4,
                            current_pos[1] - 4,
                            current_pos[0] + 4,
                            current_pos[1] + 4,
                        ),
                        fill=(255, 255, 255, 255),
                        outline=(0, 255, 0, 255),
                        width=2,
                    )

        # Add speed scale indicators
        scale_font_size = max(12, self.fontsize // 4)
        try:
            scale_font = ImageFont.truetype(
                "font/DejaVuSansCondensed-Bold.ttf", size=scale_font_size
            )
        except:
            scale_font = ImageFont.load_default()

        # Draw max speed at top
        draw.text(
            (graph_x + self.speed_graph_width + 8, graph_y - 5),
            f"{int(float(self.maxspeed))}",
            font=scale_font,
            fill=(200, 200, 200, 200),
        )

        # Draw 0 speed at bottom
        draw.text(
            (
                graph_x + self.speed_graph_width + 8,
                graph_y + self.speed_graph_height - 15,
            ),
            "0",
            font=scale_font,
            fill=(200, 200, 200, 200),
        )

        # Draw units
        draw.text(
            (
                graph_x + self.speed_graph_width + 8,
                graph_y + self.speed_graph_height // 2,
            ),
            "km/h",
            font=scale_font,
            fill=(150, 150, 150, 150),
        )

    def calcPolygonScaling(self):
        # whats the difference in longitude?
        self.lat = float(
            (int(self.maxlatitude * 10000000) - int(self.minlatitude * 10000000))
            / 10000000
        )
        self.lon = float(
            (int(self.maxlongitude * 10000000) - int(self.minlongitude * 10000000))
            / 10000000
        )
        logging.info(" lat and lon: %s %s", self.lat, self.lon)

        # which is the bigger dimension - ie how do we scale?
        if self.lat > self.lon:
            self.trackheight = int(500 * self.lon / self.lat)
            self.trackwidth = 500
        else:
            self.trackheight = 500
            self.trackwidth = int(500 * self.lat / self.lon)

        self.latscaling = int(self.trackwidth - self.polygonmargin * 2) / self.lat
        self.lonscaling = int(self.trackheight - self.polygonmargin * 2) / self.lon

        logging.info("latscaling: %s ", self.latscaling)
        logging.info("lonscaling: %s ", self.lonscaling)
        logging.info("track width x y %s %s", self.trackwidth, self.trackheight)

        self.correctedminlat = int(self.minlatitude * 10000000)
        self.correctedminlon = int(self.minlongitude * 10000000)

    def calcCoordinates(self, row):
        # logger.info("lat and lon min %s %s", self.minlatitude, self.minlongitude)

        lat = row["Latitude"]
        lon = row["Longitude"]

        # logger.info("lat and lon in %s %s", lat, lon)

        lat = int(float(lat) * 10000000)
        lon = int(float(lon) * 10000000)

        # logger.info("lat and lon float %s %s", lat, lon)

        lat = int(
            (lat - self.correctedminlat) * self.latscaling / 10000000
            + self.polygonmargin
        )
        lon = int(
            (lon - self.correctedminlon) * self.lonscaling / 10000000
            + self.polygonmargin
        )

        # logger.info("lat and lon corrected %s %s", lat, lon)
        # polygon.append((int(lat), int(lon)))
        logger.debug(
            "x y %s %s, lat lon %s %s",
            self.trackwidth,
            self.trackheight,
            lat,
            lon,
        )
        return (lat, lon)

    def genTrackPolygon(self):
        self.calcPolygonScaling()

        # collect x,y coordinates for all rows to draw polygon
        polygon = ()
        for row in self.rows:
            if row["Lap"] != "0":

                polygon = polygon + self.calcCoordinates(row)
            #        logger.info("polygon xy: %s", polygon)

        self.trackimage = self.generate_trackimage(polygon)

    def generate_trackimage(self, polygon):
        trackimage = Image.new(
            "RGBA",
            (int(self.trackwidth), int(self.trackheight)),
            (255, 255, 255, 0),
        )

        draw = ImageDraw.Draw(trackimage)
        draw.line(
            polygon,
            width=int(self.polygonwidth * 2),
            fill=(255, 255, 255, 75),
            joint="curve",
        )
        draw.line(polygon, width=self.polygonwidth, fill=(0, 0, 0), joint="curve")
        return trackimage

    def draw_position(self, row, canvas):
        coords = self.calcCoordinates(row)
        coords = (
            coords[0] - round(self.positionsize / 2),
            coords[1] - round(self.positionsize / 2),
            coords[0] + round(self.positionsize / 2),
            coords[1] + round(self.positionsize / 2),
        )
        img = canvas.copy()
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(
            coords, fill=(255, 0, 0), outline=None, width=5, radius=2
        )

        return img

    def thread_worker(self, q):
        while True:
            row = q.get()
            self.generate_image(row=row)
            q.task_done()

    async def generate_images(self):
        q = queue.Queue()

        processcount = 0
        while processcount < cpu_count():
            logger.info("starting worker %s of %s", processcount + 1, cpu_count())
            threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
            processcount = processcount + 1

        logger.info("all threads started")
        self.log.append("all threads started")

        for row in self.rows:
            q.put(row)

        logger.info("all " + str(len(self.rows)) + " rows added to queue")
        self.log.append("all " + str(len(self.rows)) + " rows added to queue")

        q.join()
        logger.info("All images built")

    async def generate_images_batched(self, batch_size=500):
        """Generate images in batches with concurrent ffmpeg processing"""
        total_rows = len(self.rows)
        total_batches = (total_rows + batch_size - 1) // batch_size

        # Semaphore to limit concurrent ffmpeg instances
        ffmpeg_semaphore = asyncio.Semaphore(2)

        # List to collect all segment paths and ffmpeg tasks
        segment_paths = []
        ffmpeg_tasks = []

        logger.info(
            f"🎬 CONCURRENT PROCESSING: {total_rows} frames in {total_batches} batches of {batch_size}"
        )
        logger.info("📸 Image generation will run continuously in background")
        logger.info("🎞️  FFmpeg will process batches as they become ready")
        logger.info("🔄 Up to 2 FFmpeg instances can run simultaneously")

        # Queue to track completed image batches ready for ffmpeg
        completed_batches = asyncio.Queue()

        async def image_generator():
            """Generate images for all batches in background"""
            logger.info("🏁 Starting background image generation...")

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_rows)
                batch_rows = self.rows[start_idx:end_idx]

                logger.info(
                    f"📸 Generating images for batch {batch_idx + 1}/{total_batches} (frames {start_idx}-{end_idx-1})"
                )

                # Generate images for this batch
                await self._generate_batch_images(batch_rows, batch_idx)

                # Signal that this batch is ready for ffmpeg
                await completed_batches.put((batch_idx, batch_rows))
                logger.info(
                    f"✅ Batch {batch_idx + 1} images ready - queued for ffmpeg"
                )

                # Yield control to allow FFmpeg processor to run
                await asyncio.sleep(0)

            # Signal completion
            await completed_batches.put(None)
            logger.info("🏁 All image generation completed")

        async def ffmpeg_processor():
            """Process ffmpeg jobs as batches become available"""
            logger.info("🎬 Starting ffmpeg processor...")
            processed_batches = 0

            try:
                while True:
                    # Wait for next batch or completion signal
                    logger.info("🎬 FFmpeg processor waiting for next batch...")
                    batch_info = await completed_batches.get()

                    if batch_info is None:
                        logger.info("🎬 FFmpeg processor: All batches completed")
                        break

                    batch_idx, batch_rows = batch_info

                    # Create video segment path
                    segment_path = f"{self.filename}_segment_{batch_idx:04d}.mov"
                    segment_paths.append(segment_path)

                    logger.info(f"🎞️  Starting ffmpeg for batch {batch_idx + 1}...")

                    # Create ffmpeg task
                    ffmpeg_task = asyncio.create_task(
                        self._create_video_segment_with_cleanup(
                            batch_idx, segment_path, ffmpeg_semaphore, batch_rows
                        )
                    )
                    ffmpeg_tasks.append(ffmpeg_task)

                    processed_batches += 1
                    logger.info(
                        f"⚡ FFmpeg task {processed_batches}/{total_batches} started"
                    )
            except Exception as e:
                logger.error(f"❌ Error in ffmpeg_processor: {e}")
                import traceback

                traceback.print_exc()
                raise

        # Start both generators concurrently
        logger.info("🔧 Creating image generation task...")
        image_task = asyncio.create_task(image_generator())
        logger.info("🔧 Creating ffmpeg processor task...")
        ffmpeg_task = asyncio.create_task(ffmpeg_processor())

        # Let both tasks run concurrently - don't wait for them to complete yet
        logger.info(
            "🚀 Both image generation and ffmpeg processing started concurrently"
        )

        # Wait for both to complete
        await asyncio.gather(image_task, ffmpeg_task)

        # Wait for all ffmpeg tasks to complete
        logger.info("Waiting for all ffmpeg tasks to complete...")
        await asyncio.gather(*ffmpeg_tasks)

        # Force garbage collection
        import gc

        gc.collect()

        logger.info("All video segments created, concatenating...")

        # Concatenate all segments
        final_output = await self._concatenate_segments(segment_paths)

        # Cleanup segment files
        for segment in segment_paths:
            if os.path.exists(segment):
                os.remove(segment)

        return final_output

    async def _generate_batch_images(self, batch_rows, batch_idx):
        """Generate images for a specific batch"""
        logger.info(
            f"Starting image generation for batch {batch_idx + 1} ({len(batch_rows)} frames)"
        )

        # Use ThreadPoolExecutor to process only this batch's rows
        # Limit to half the CPU cores to avoid memory overload
        max_workers = max(1, cpu_count() // 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch rows for processing
            futures = [executor.submit(self.generate_image, row) for row in batch_rows]

            # Wait for all images in this batch to complete
            for future in futures:
                future.result()  # This will raise an exception if image generation failed

        logger.info(f"BATCH {batch_idx + 1} IMAGES COMPLETED - ready for ffmpeg")

    async def _create_video_segment(
        self, batch_idx, segment_path, semaphore, batch_rows
    ):
        """Create a video segment from a batch of images"""
        logger.info(f"Waiting for ffmpeg slot for batch {batch_idx + 1}...")
        async with semaphore:
            logger.info(
                f"FFMPEG STARTED: Creating video segment {batch_idx + 1}: {segment_path}"
            )

            try:
                # Count available frames for this batch
                frame_count = 0
                for row in batch_rows:
                    frame_num = int(row["Record"])
                    frame_file = f"{self.foldername}frame{frame_num:08d}.png"
                    if os.path.exists(frame_file):
                        frame_count += 1
                    else:
                        logger.warning(f"Frame file missing: {frame_file}")

                if frame_count == 0:
                    logger.error(
                        f"Batch {batch_idx + 1}: No frames found! Skipping ffmpeg."
                    )
                    return

                logger.info(
                    f"Batch {batch_idx + 1}: Processing {frame_count} frames with ffmpeg"
                )

                # Use pattern-based input like the original working method
                # Create a temporary directory with only this batch's frames
                batch_dir = f"{self.tmpdir.name}/batch_{batch_idx}"
                os.makedirs(batch_dir, exist_ok=True)

                # Copy/link only the frames for this batch
                for row in batch_rows:
                    frame_num = int(row["Record"])
                    src_frame = f"{self.foldername}frame{frame_num:08d}.png"
                    dst_frame = f"{batch_dir}/frame{frame_num:08d}.png"
                    if os.path.exists(src_frame):
                        os.link(src_frame, dst_frame)  # Hard link to save space

                batch_pattern = f"{batch_dir}/frame*.png"

                cmd = (
                    ffmpeg.input(batch_pattern, pattern_type="glob", framerate=25)
                    .output(
                        segment_path,
                        vcodec="prores_ks",
                        pix_fmt="yuva444p10le",
                        qscale=4,
                    )
                    .overwrite_output()
                )
                logger.info(f"FFmpeg command: {' '.join(cmd.compile())}")

                await asyncio.get_event_loop().run_in_executor(None, lambda: cmd.run())
                logger.info(
                    f"FFMPEG COMPLETED: Video segment {batch_idx + 1}: {segment_path}"
                )

                # Clean up batch directory
                import shutil

                shutil.rmtree(batch_dir)

            except Exception as e:
                logger.error(f"Error creating video segment {batch_idx + 1}: {e}")
                # Clean up batch directory on error
                batch_dir = f"{self.tmpdir.name}/batch_{batch_idx}"
                if os.path.exists(batch_dir):
                    import shutil

                    shutil.rmtree(batch_dir)
                raise

    async def _create_video_segment_with_cleanup(
        self, batch_idx, segment_path, semaphore, batch_rows
    ):
        """Create a video segment from a batch of images and clean up afterward"""
        try:
            # Create the video segment
            await self._create_video_segment(
                batch_idx, segment_path, semaphore, batch_rows
            )

            # Clean up the original image files for this batch to save memory
            logger.info(f"Cleaning up image files for batch {batch_idx + 1}...")
            for row in batch_rows:
                frame_num = int(row["Record"])
                frame_file = f"{self.foldername}frame{frame_num:08d}.png"
                if os.path.exists(frame_file):
                    os.remove(frame_file)

            logger.info(f"Batch {batch_idx + 1} processing and cleanup completed")

        except Exception as e:
            logger.error(f"Error in batch {batch_idx + 1} processing: {e}")
            raise

    async def _concatenate_segments(self, video_segments):
        """Concatenate all video segments into final output"""
        final_output = self.filename + ".mov"

        # Create concat file list
        concat_file = f"{self.tmpdir.name}/concat_list.txt"
        with open(concat_file, "w") as f:
            for segment in video_segments:
                if os.path.exists(segment):
                    f.write(f"file '{os.path.abspath(segment)}'\n")
                else:
                    logger.warning(f"Segment file not found: {segment}")

        # Get the start timecode from CSV
        start_timecode = self._get_start_timecode()

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: (
                    ffmpeg.input(concat_file, format="concat", safe=0)
                    .output(final_output, c="copy", timecode=start_timecode)
                    .overwrite_output()
                    .run()
                ),
            )
            logger.info(f"Final video created: {final_output}")
            self.tmpdir.cleanup()
            return final_output
        except Exception as e:
            logger.error(f"Error concatenating segments: {e}")
            raise

    async def generate_movie(self):
        """Generate movie using incremental batch processing"""
        return await self.generate_images_batched()

    async def generate_movie_legacy(self):
        """Original movie generation method (kept for reference)"""
        pass1 = (
            ffmpeg.input(self.foldername + "*.png", pattern_type="glob", framerate=25)
            .output(
                self.filename + ".mov",
                vcodec="prores_ks",
                pix_fmt="yuva444p10le",
                qscale=4,
            )
            .overwrite_output()
            .run()
        )
        self.log.append(pass1)
        self.tmpdir.cleanup()
        return self.filename + ".mov"

    def generate_textbox(
        self,
        draw,
        x,
        y,
        text,
        font=None,
        color=(255, 255, 255, 200),
        align="center",
    ):
        if font is None:
            font = self.font
        draw_point = (x, y + 6)
        length = draw.textlength(text, font=font)
        draw.rounded_rectangle(
            xy=(
                (x, y + self.fontsize * 0.2),
                (x + length, y + self.fontsize + self.fontsize * 0.2),
            ),
            radius=5,
            fill=(50, 50, 50, 80),
            outline=None,
        )
        draw.text(draw_point, text, font=font, fill=color, align=align)

    def generate_image(self, row=dict):
        frame = row["Record"]
        speed = row["Speed"]
        speed = speed[:-3] + " km/h"
        lap = "Lap: " + row["Lap"]
        lean = float(row["LeanAngle"])
        orglean = int(round(lean))
        if lean > 0:
            lean = str(round(lean))
        else:
            lean = str(round(-lean))

        gforcez = "gforcez" + str(row["GForceZ"])
        gyrox = "gyrox" + str(row["GyroX"])
        gyroy = "gyroy" + str(row["GyroY"])
        gyroz = "gyroz" + str(row["GyroZ"])

        lean = lean + "°"

        gforce = row["GForceX"]

        # Generate filename with frame number padded to 8 digits
        filename = self.foldername + "frame" + str(int(frame)).rjust(8, "0") + ".png"

        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        draw = ImageDraw.Draw(img)

        if int(row["Lap"]) > 0:
            self.generate_textbox(
                draw=draw,
                x=self.width * 0.08,
                y=self.height * 0.10 + self.fontsize * 2 + self.fontsize * 0.15 * 2,
                text=lap,
                align="left",
            )

        # Show lap times after completing at least one lap (keep them visible even in lap 0)
        if int(row["Lap"]) > 1 or self.hasShownLaps:
            lapcnt = 1
            self.hasShownLaps = True
            current_lap_for_display = int(row["Lap"])
            if current_lap_for_display == 0:
                # make sure we show laptimes for the in-lap
                current_lap_for_display = 999
            for lap in self.laps:
                if lapcnt < current_lap_for_display:
                    if current_lap_for_display > 2:
                        if lap <= self.prevBest:
                            self.prevBest = lap
                            color = (30, 255, 30, 200)
                        else:
                            color = (255, 255, 255, 200)
                    else:
                        color = (255, 255, 255, 200)

                    self.generate_textbox(
                        draw=draw,
                        x=self.width * 0.08,
                        y=self.height * 0.10
                        + self.fontsize * 2
                        + self.fontsize * 0.15 * 2
                        + self.fontsize * 0.15 * lapcnt
                        + self.fontsize * lapcnt,
                        # Prepare lap time formatting with zero-padded seconds
                        text="Lap "
                        + str(lapcnt)
                        + ": "
                        + str(self.convert_seconds(int(str(lap).split(".")[0]))[0])
                        + ":"
                        + str(
                            self.convert_seconds(int(str(lap).split(".")[0]))[1]
                        ).zfill(2)
                        + "."
                        + str(lap).split(".")[1],
                        align="left",
                        color=color,
                    )
                lapcnt = lapcnt + 1

        self.generate_textbox(
            draw=draw,
            x=self.width * 0.08,
            y=self.height * 0.8,
            text=speed,
        )

        # Draw speed graph above the speed indicator
        self.draw_speed_graph(draw, row)

        drawimage = self.draw_position(row, self.trackimage)

        img.paste(
            drawimage,
            (self.width - int(self.trackwidth) - int(self.trackwidth * 1), 50),
        )

        overlay = self.arrow_images.get(orglean)
        if overlay:
            img.paste(
                overlay,
                (int(self.width * 0.8), int(self.height * 0.74)),
                overlay,
            )

        self.generate_textbox(
            draw=draw, x=self.width * 0.832, y=self.height * 0.74, text=lean
        )

        # Only show G meter if we have sufficient data variation (both positive and negative forces)
        if self.show_g_meter:
            if float(gforce) > 0:
                gtext = "⬇" + str(round(float(gforce), 2)) + "G"
                gcolor = (255, 20, 20, 200)
            else:
                gtext = "⬆" + str(round(-float(gforce), 2)) + "G"
                gcolor = (20, 255, 20, 200)

            self.generate_textbox(
                draw=draw,
                x=self.width * 0.8,
                y=self.height * 0.8,
                text=gtext,
                color=gcolor,
                align="right",
            )

        img.save(filename, "PNG", compress_level=1)
        # logger.info("Saved file " + filename)
        self.log.append("Saved file " + filename)
