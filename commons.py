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

        # Initialize arrow_images dictionary
        self.arrow_images = {}
        for angle in range(-90, 91):
            path = f"angles/arrow_{angle:03d}.png"
            if os.path.exists(path):
                self.arrow_images[angle] = Image.open(path).convert("RGBA")

    def convert_seconds(self, seconds):
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return minutes, remaining_seconds

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

    async def generate_images_batched(self, batch_size=1000):
        """Generate images in batches and create video segments incrementally"""
        total_rows = len(self.rows)
        total_batches = (total_rows + batch_size - 1) // batch_size

        # Semaphore to limit concurrent ffmpeg instances
        ffmpeg_semaphore = asyncio.Semaphore(2)

        # List to collect all segment paths and tasks
        segment_tasks = []
        segment_paths = []

        logger.info(
            f"Processing {total_rows} frames in {total_batches} batches of {batch_size}"
        )

        # Create a task for each batch that will:
        # 1. Generate images for the batch
        # 2. Then immediately start ffmpeg for that batch
        async def process_batch(batch_idx):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_rows = self.rows[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches} (frames {start_idx}-{end_idx-1})"
            )

            # Generate images for this batch
            await self._generate_batch_images(batch_rows, batch_idx)

            # Create video segment path
            segment_path = f"{self.filename}_segment_{batch_idx:04d}.mov"
            segment_paths.append(segment_path)

            logger.info(f"Batch {batch_idx + 1} images ready, starting ffmpeg...")

            # Start ffmpeg for this batch immediately
            await self._create_video_segment(
                batch_idx, segment_path, ffmpeg_semaphore, batch_rows
            )

        # Start all batch processing tasks concurrently
        batch_tasks = [process_batch(batch_idx) for batch_idx in range(total_batches)]
        await asyncio.gather(*batch_tasks)

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
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
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

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: (
                    ffmpeg.input(concat_file, format="concat", safe=0)
                    .output(final_output, c="copy")
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

        date = row["Time"].split("T")
        time = date[1].split(".")

        gforce = row["GForceX"]

        # Generate filename with frame number padded to 8 digits
        filename = self.foldername + "frame" + str(int(frame)).rjust(8, "0") + ".png"

        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        draw = ImageDraw.Draw(img)

        # left top
        self.generate_textbox(
            draw=draw,
            x=self.width * 0.08,
            y=self.height * 0.10,
            text=date[0],
            align="left",
        )

        self.generate_textbox(
            draw=draw,
            x=self.width * 0.08,
            y=self.height * 0.10 + self.fontsize * 1.15,
            text=time[0],
            align="left",
        )

        if int(row["Lap"]) > 0:
            self.generate_textbox(
                draw=draw,
                x=self.width * 0.08,
                y=self.height * 0.10 + self.fontsize * 2 + self.fontsize * 0.15 * 2,
                text=lap,
                align="left",
            )

        if int(row["Lap"]) > 1 or self.hasShownLaps is True:
            lapcnt = 1
            self.hasShownLaps = True
            if row["Lap"] == "0":
                # make sure we show laptimes for the in-lap
                row["Lap"] = 999
            for lap in self.laps:
                if lapcnt < int(row["Lap"]):
                    if int(row["Lap"]) > 2:
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
                        # Prepare lap time formatting
                        text="Lap "
                        + str(lapcnt)
                        + ": "
                        + str(self.convert_seconds(int(str(lap).split(".")[0]))[0])
                        + ":"
                        + str(self.convert_seconds(int(str(lap).split(".")[0]))[1])
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
