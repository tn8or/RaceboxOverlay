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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from multiprocessing import cpu_count

import ffmpeg
from PIL import Image, ImageDraw, ImageFont

from database import LapTimeDatabase


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


def generate_single_image_worker(row_data, config_data):
    """
    Standalone function for generating a single image in a separate process.

    Args:
        row_data: Dictionary containing the CSV row data
        config_data: Dictionary containing all necessary configuration data
    """
    try:
        # Extract configuration
        foldername = config_data['foldername']
        width = config_data['width']
        height = config_data['height']
        fontsize = config_data['fontsize']
        maxspeed = config_data['maxspeed']
        show_g_meter = config_data['show_g_meter']
        maxaccg = config_data['maxaccg']
        maxdecg = config_data['maxdecg']

        # Import here to avoid issues with multiprocessing
        from PIL import Image, ImageDraw, ImageFont
        import os

        # Generate image using the same logic as generate_image method
        frame = row_data["Record"]
        speed = row_data["Speed"]
        speed = speed[:-3] + " km/h"
        lean = float(row_data["LeanAngle"])
        orglean = int(round(lean))
        if lean > 0:
            lean = str(round(lean))
        else:
            lean = str(round(-lean))
        lean = lean + "°"
        gforce = row_data["GForceX"]

        # Generate filename with frame number padded to 8 digits
        filename = foldername + "frame" + str(int(frame)).rjust(8, "0") + ".png"

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Load font
        try:
            font = ImageFont.truetype("font/DejaVuSansCondensed-Bold.ttf", size=fontsize)
        except Exception:
            font = ImageFont.load_default()

        # Simple speed indicator (minimal implementation for performance)
        draw_point = (width * 0.08, height * 0.8 + 6)
        length = draw.textlength(speed, font=font)
        draw.rounded_rectangle(
            xy=((width * 0.08, height * 0.8 + fontsize * 0.2),
                (width * 0.08 + length, height * 0.8 + fontsize + fontsize * 0.2)),
            radius=5,
            fill=(50, 50, 50, 80),
            outline=None,
        )
        draw.text(draw_point, speed, font=font, fill=(255, 255, 255, 200))

        # Add G-force display if enabled
        if show_g_meter and (maxaccg > 0.1 and maxdecg > 0.1):
            if float(gforce) != 0:
                if float(gforce) < 0:
                    gtext = "⬇" + str(round(float(gforce), 2)) + "G"
                    gcolor = (255, 20, 20, 200)
                else:
                    gtext = "⬆" + str(round(-float(gforce), 2)) + "G"
                    gcolor = (20, 255, 20, 200)

                # G-force text
                g_draw_point = (width * 0.8, height * 0.8 + 6)
                g_length = draw.textlength(gtext, font=font)
                draw.rounded_rectangle(
                    xy=((width * 0.8 - g_length, height * 0.8 + fontsize * 0.2),
                        (width * 0.8, height * 0.8 + fontsize + fontsize * 0.2)),
                    radius=5,
                    fill=(50, 50, 50, 80),
                    outline=None,
                )
                draw.text((width * 0.8 - g_length, height * 0.8 + 6), gtext, font=font, fill=gcolor)

        # Save the image
        img.save(filename, "PNG", compress_level=1)
        return filename

    except Exception as e:
        # Use print instead of logger since this runs in a separate process
        print(f"Error generating image for frame {row_data.get('Record', 'unknown')}: {e}")
        raise


def _generate_image_worker_process(row, width, height, fontsize, foldername):
    """
    Standalone function for process-based image generation.
    This function recreates the necessary context in each process to generate images.
    NOTE: This is a simplified version for demonstration. For full functionality,
    we need to pass more class data or use threading instead of processes.
    """
    from PIL import Image, ImageDraw, ImageFont
    import logging

    # Set up logger for this process
    logger = logging.getLogger(__name__)

    try:
        # Create new image canvas in this process
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Extract row data
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
        lean = lean + "°"

        # Generate filename with frame number padded to 8 digits
        filename = foldername + "frame" + str(int(frame)).rjust(8, "0") + ".png"

        draw = ImageDraw.Draw(img)

        # Create font for this process
        try:
            font = ImageFont.truetype("font/DejaVuSansCondensed-Bold.ttf", size=fontsize)
        except:
            font = ImageFont.load_default()

        # Generate simple textbox function for this process
        def generate_textbox_process(draw, x, y, text, font, color=(255, 255, 255, 200)):
            draw_point = (x, y + 6)
            length = draw.textlength(text, font=font)
            draw.rounded_rectangle(
                xy=(
                    (x, y + fontsize * 0.2),
                    (x + length, y + fontsize + fontsize * 0.2),
                ),
                radius=5,
                fill=(50, 50, 50, 80),
                outline=None,
            )
            draw.text(draw_point, text, font=font, fill=color, align="center")

        # Draw lap number (simplified - just current lap)
        current_lap_number = int(row["Lap"])
        if current_lap_number > 0:
            generate_textbox_process(
                draw=draw,
                x=width * 0.08,
                y=50,
                text=lap,
                font=font,
            )

        # Draw speed indicator
        generate_textbox_process(
            draw=draw,
            x=width * 0.08,
            y=height * 0.8,
            text=speed,
            font=font,
        )

        # Draw lean angle indicator
        generate_textbox_process(
            draw=draw,
            x=width * 0.832,
            y=height * 0.74,
            text=lean,
            font=font
        )

        # TODO: Add track map, sector times, speed graph, etc.
        # These require class data that's not easily passed to processes
        # For full functionality, consider using threading instead

        # Save the image
        img.save(filename)

        return True

    except Exception as e:
        logger.error(f"Error in process worker for frame {row.get('Record', 'unknown')}: {e}")
        return False


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
        self.last_lap_number = 0  # Track the last lap number for lap timer updates

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

        # Parse lap times and sector times from header
        self.sector_times = {}  # {lap_number: [sector1, sector2, ...]}
        self.real_time_sectors = (
            {}
        )  # Track sectors as they complete during video generation
        self.completed_sectors = {}  # Track which sectors are completed for each lap
        for row in header:
            if row.startswith("Lap "):
                lap_parts = row.split(",")
                if len(lap_parts) > 1:
                    lap_time = float(lap_parts[1].strip())
                    if lap_time not in self.laps:
                        self.laps.append(lap_time)

                    # Extract lap number and sector times
                    lap_number = int(lap_parts[0].split()[1])
                    sectors = []
                    if len(lap_parts) > 3 and lap_parts[2].strip() == "sectors":
                        # Parse sector times (skip "sectors" and take the numeric values)
                        for i in range(3, len(lap_parts)):
                            try:
                                sector_time = float(lap_parts[i].strip())
                                if sector_time > 0:  # Only include valid sector times
                                    sectors.append(sector_time)
                            except ValueError:
                                continue
                    self.sector_times[lap_number] = sectors

        # Extract track information from header
        self.track_name = "Unknown Track"
        self.session_date = "Unknown Date"
        self.date_utc = ""

        for row in header:
            if row.startswith("Track,"):
                self.track_name = row.split(",")[1].strip()
            elif row.startswith("Date UTC,"):
                self.date_utc = row.split(",")[1].strip()
            elif row.startswith("Date,"):
                self.session_date = row.split(",")[1].strip()

        # Initialize database
        self.db = LapTimeDatabase()
        self.session_id = self.db.get_or_create_session(
            self.date_utc, self.track_name, self.session_date
        )

        # Store lap and sector data in database
        for lap_num, lap_time in enumerate(self.laps, 1):
            sectors = self.sector_times.get(lap_num, [])
            self.db.store_lap_with_sectors(self.session_id, lap_num, lap_time, sectors)

        logger.info("found %s laps", len(self.laps))
        logger.info("laps: %s", self.laps)
        logger.info("Track: %s, Session: %s", self.track_name, self.session_date)
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

    def get_sector_counter_text(self, current_row):
        """Generate sector counter display with actual times as sectors complete"""
        current_lap = int(current_row["Lap"])
        if current_lap <= 0:
            return ""

        current_frame = int(current_row["Record"])

        # Find total frames in current lap to calculate progress
        lap_frames = []
        for row in self.rows:
            if int(row["Lap"]) == current_lap:
                lap_frames.append(int(row["Record"]))

        if not lap_frames:
            return ""

        lap_start = min(lap_frames)
        lap_end = max(lap_frames)
        total_frames = lap_end - lap_start if lap_end > lap_start else 1
        current_progress = (current_frame - lap_start) / total_frames

        # Initialize sectors for this lap if not exists
        if current_lap not in self.real_time_sectors:
            self.real_time_sectors[current_lap] = {}
            self.completed_sectors[current_lap] = set()

        # Calculate sector times based on progress (assuming 3 equal sectors)
        sector_1_progress = 0.33
        sector_2_progress = 0.66

        current_lap_time = self.get_current_lap_time(current_row)

        # Build sector display text
        sector_display = []

        if (
            current_progress >= sector_1_progress
            and 1 not in self.completed_sectors[current_lap]
        ):
            # Just completed sector 1
            sector_1_time = current_lap_time * sector_1_progress
            self.real_time_sectors[current_lap][1] = sector_1_time
            self.completed_sectors[current_lap].add(1)
            # Store in database for real-time color updates
            try:
                self.db.store_sector(self.session_id, current_lap, 1, sector_1_time)
            except:
                pass  # May already exist

        if (
            current_progress >= sector_2_progress
            and 2 not in self.completed_sectors[current_lap]
        ):
            # Just completed sector 2
            sector_1_time = self.real_time_sectors[current_lap].get(1, 0)
            sector_2_time = current_lap_time * sector_2_progress - sector_1_time
            self.real_time_sectors[current_lap][2] = sector_2_time
            self.completed_sectors[current_lap].add(2)
            # Store in database for real-time color updates
            try:
                self.db.store_sector(self.session_id, current_lap, 2, sector_2_time)
            except:
                pass  # May already exist

        if current_progress >= 1.0 and 3 not in self.completed_sectors[current_lap]:
            # Just completed sector 3 (lap finished)
            sector_1_time = self.real_time_sectors[current_lap].get(1, 0)
            sector_2_time = self.real_time_sectors[current_lap].get(2, 0)
            sector_3_time = current_lap_time - sector_1_time - sector_2_time
            self.real_time_sectors[current_lap][3] = sector_3_time
            self.completed_sectors[current_lap].add(3)
            # Store in database for real-time color updates
            try:
                self.db.store_sector(self.session_id, current_lap, 3, sector_3_time)
            except:
                pass  # May already exist

        # Build display text with completed sector times
        if 1 in self.completed_sectors[current_lap]:
            s1_time = self.real_time_sectors[current_lap][1]
            sector_display.append(f"S1: {s1_time:.3f}")
        else:
            sector_display.append("S1:")

        if 2 in self.completed_sectors[current_lap]:
            s2_time = self.real_time_sectors[current_lap][2]
            sector_display.append(f"S2: {s2_time:.3f}")
        elif 1 in self.completed_sectors[current_lap]:
            sector_display.append("S2:")

        if 3 in self.completed_sectors[current_lap]:
            s3_time = self.real_time_sectors[current_lap][3]
            sector_display.append(f"S3: {s3_time:.3f}")
        elif 2 in self.completed_sectors[current_lap]:
            sector_display.append("S3:")

        return "  ".join(sector_display)

    def draw_sector_counter_with_colors(self, draw, current_row, x, y):
        """Draw sector counter with individual colors for each sector"""
        current_lap = int(current_row["Lap"])
        if current_lap <= 0:
            return

        current_frame = int(current_row["Record"])

        # Find total frames in current lap to calculate progress
        lap_frames = []
        for row in self.rows:
            if int(row["Lap"]) == current_lap:
                lap_frames.append(int(row["Record"]))

        if not lap_frames:
            return

        lap_start = min(lap_frames)
        lap_end = max(lap_frames)
        total_frames = lap_end - lap_start if lap_end > lap_start else 1
        current_progress = (current_frame - lap_start) / total_frames

        # Initialize sectors for this lap if not exists
        if current_lap not in self.real_time_sectors:
            self.real_time_sectors[current_lap] = {}
            self.completed_sectors[current_lap] = set()

        # Calculate sector times based on progress (assuming 3 equal sectors)
        sector_1_progress = 0.33
        sector_2_progress = 0.66

        current_lap_time = self.get_current_lap_time(current_row)

        # Update completed sectors based on progress
        if (
            current_progress >= sector_1_progress
            and 1 not in self.completed_sectors[current_lap]
        ):
            sector_1_time = current_lap_time * sector_1_progress
            self.real_time_sectors[current_lap][1] = sector_1_time
            self.completed_sectors[current_lap].add(1)
            try:
                self.db.store_sector(self.session_id, current_lap, 1, sector_1_time)
            except:
                pass

        if (
            current_progress >= sector_2_progress
            and 2 not in self.completed_sectors[current_lap]
        ):
            sector_1_time = self.real_time_sectors[current_lap].get(1, 0)
            sector_2_time = current_lap_time * sector_2_progress - sector_1_time
            self.real_time_sectors[current_lap][2] = sector_2_time
            self.completed_sectors[current_lap].add(2)
            try:
                self.db.store_sector(self.session_id, current_lap, 2, sector_2_time)
            except:
                pass

        if current_progress >= 1.0 and 3 not in self.completed_sectors[current_lap]:
            sector_1_time = self.real_time_sectors[current_lap].get(1, 0)
            sector_2_time = self.real_time_sectors[current_lap].get(2, 0)
            sector_3_time = current_lap_time - sector_1_time - sector_2_time
            self.real_time_sectors[current_lap][3] = sector_3_time
            self.completed_sectors[current_lap].add(3)
            try:
                self.db.store_sector(self.session_id, current_lap, 3, sector_3_time)
            except:
                pass

        # Draw each sector with its own color
        current_x = x

        # Create font for text measurement
        try:
            font = ImageFont.truetype(
                "font/DejaVuSansCondensed-Bold.ttf", size=self.fontsize
            )
        except Exception:
            font = ImageFont.load_default()

        # Sector 1
        if 1 in self.completed_sectors[current_lap]:
            # Use actual CSV sector time for color determination if available
            if (
                current_lap in self.sector_times
                and len(self.sector_times[current_lap]) >= 1
            ):
                s1_time = self.sector_times[current_lap][0]  # First sector from CSV
            else:
                s1_time = self.real_time_sectors[current_lap][
                    1
                ]  # Fallback to calculated
            s1_text = f"S1: {s1_time:.3f}"
            s1_color = self.get_sector_color(current_lap, 1, s1_time)
        else:
            s1_text = "S1:"
            s1_color = (200, 200, 200, 180)  # Gray for incomplete

        self.generate_textbox(
            draw=draw,
            x=current_x,
            y=y,
            text=s1_text,
            align="left",
            color=s1_color,
        )
        # Use proper text width for spacing
        s1_width = draw.textlength(s1_text, font=font)
        current_x += s1_width + 25  # Proper spacing between sectors

        # Sector 2 (only show if sector 1 is completed or sector 2 is in progress)
        if 2 in self.completed_sectors[current_lap]:
            # Use actual CSV sector time for color determination if available
            if (
                current_lap in self.sector_times
                and len(self.sector_times[current_lap]) >= 2
            ):
                s2_time = self.sector_times[current_lap][1]  # Second sector from CSV
            else:
                s2_time = self.real_time_sectors[current_lap][
                    2
                ]  # Fallback to calculated
            s2_text = f"S2: {s2_time:.3f}"
            s2_color = self.get_sector_color(current_lap, 2, s2_time)
        elif 1 in self.completed_sectors[current_lap]:
            s2_text = "S2:"
            s2_color = (200, 200, 200, 180)  # Gray for incomplete
        else:
            s2_text = ""

        if s2_text:
            self.generate_textbox(
                draw=draw,
                x=current_x,
                y=y,
                text=s2_text,
                align="left",
                color=s2_color,
            )
            # Use proper text width for spacing
            s2_width = draw.textlength(s2_text, font=font)
            current_x += s2_width + 25  # Proper spacing between sectors

        # Sector 3 (only show if sector 2 is completed or sector 3 is in progress)
        if 3 in self.completed_sectors[current_lap]:
            # Use actual CSV sector time for color determination if available
            if (
                current_lap in self.sector_times
                and len(self.sector_times[current_lap]) >= 3
            ):
                s3_time = self.sector_times[current_lap][2]  # Third sector from CSV
            else:
                s3_time = self.real_time_sectors[current_lap][
                    3
                ]  # Fallback to calculated
            s3_text = f"S3: {s3_time:.3f}"
            s3_color = self.get_sector_color(current_lap, 3, s3_time)
        elif 2 in self.completed_sectors[current_lap]:
            s3_text = "S3:"
            s3_color = (200, 200, 200, 180)  # Gray for incomplete
        else:
            s3_text = ""

        if s3_text:
            self.generate_textbox(
                draw=draw,
                x=current_x,
                y=y,
                text=s3_text,
                align="left",
                color=s3_color,
            )

    def convert_seconds(self, seconds):
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return minutes, remaining_seconds

    def get_current_lap_time(self, current_row):
        """Calculate elapsed time for current lap"""
        current_lap = int(current_row["Lap"])
        if current_lap == 0:
            return 0.0

        current_frame = int(current_row["Record"])

        # Find the start frame of the current lap
        lap_start_frame = None
        for row in self.rows:
            if int(row["Lap"]) == current_lap:
                if lap_start_frame is None:
                    lap_start_frame = int(row["Record"])
                break

        if lap_start_frame is None:
            return 0.0

        # Calculate elapsed frames and convert to seconds (assuming 25fps)
        elapsed_frames = current_frame - lap_start_frame
        elapsed_seconds = elapsed_frames / 25.0  # 25 fps
        return elapsed_seconds

    def get_lap_color(
        self, lap_number: int, lap_time: float, context_lap: int = None
    ) -> tuple:
        """
        Determine color for lap time based on fastest so far up to context_lap.

        Args:
            lap_number: The lap whose time we're coloring
            lap_time: The time for that lap
            context_lap: The current lap being processed (for "point in time" context)
                        If None, uses lap_number as context
        """
        try:
            # Use context_lap to determine what laps to consider for "fastest so far"
            current_context = context_lap if context_lap is not None else lap_number

            # Get fastest lap up to the current context
            fastest_lap_number = self.get_fastest_lap_number_so_far(current_context)

            # This lap is purple if it's currently the fastest so far
            is_currently_fastest = fastest_lap_number == lap_number

            if is_currently_fastest:
                return (128, 0, 128, 255)  # Purple - currently fastest so far
            else:
                return (255, 255, 255, 200)  # White - normal or previously fastest
        except Exception as e:
            logger.warning("Error determining lap color: %s", e)
            return (255, 255, 255, 200)

    def get_fastest_lap_number_so_far(self, current_lap: int) -> int:
        """Get the lap number that has the fastest time up to and including current_lap"""
        fastest_time = None
        fastest_lap_num = None

        # Look through completed laps up to current lap
        for lap_num in range(1, current_lap + 1):
            if lap_num <= len(self.laps):
                lap_time = self.laps[lap_num - 1]
                if fastest_time is None or lap_time < fastest_time:
                    fastest_time = lap_time
                    fastest_lap_num = lap_num

        return fastest_lap_num

    def get_fastest_lap_so_far(self, current_lap: int) -> float:
        """Get the fastest lap time up to and including the current lap"""
        fastest_time = None

        # Look through completed laps up to current lap
        for lap_num in range(1, current_lap + 1):
            if lap_num in self.sector_times:
                # Use lap time from CSV header if available
                lap_time = self.laps[lap_num - 1] if lap_num <= len(self.laps) else None
                if lap_time and (fastest_time is None or lap_time < fastest_time):
                    fastest_time = lap_time

        return fastest_time

    def get_fastest_sector_so_far(self, current_lap: int, sector_number: int) -> float:
        """Get the fastest sector time up to and including the current lap"""
        fastest_time = None

        # Look through completed laps up to current lap
        for lap_num in range(1, current_lap + 1):
            if (
                lap_num in self.sector_times
                and len(self.sector_times[lap_num]) >= sector_number
            ):
                sector_time = self.sector_times[lap_num][sector_number - 1]  # 0-indexed
                if fastest_time is None or sector_time < fastest_time:
                    fastest_time = sector_time

        return fastest_time

    def get_sector_color(
        self,
        lap_number: int,
        sector_number: int,
        sector_time: float,
        context_lap: int = None,
    ) -> tuple:
        """
        Determine color for sector time based on fastest so far up to context_lap.

        Args:
            lap_number: The lap whose sector time we're coloring
            sector_number: Which sector (1, 2, or 3)
            sector_time: The time for that sector
            context_lap: The current lap being processed (for "point in time" context)
                        If None, uses lap_number as context
        """
        try:
            # Use context_lap to determine what laps to consider for "fastest so far"
            current_context = context_lap if context_lap is not None else lap_number

            # Get the lap that has the fastest sector up to the current context
            fastest_lap_number = self.get_fastest_sector_lap_number_so_far(
                current_context, sector_number
            )

            # This sector is purple if it's currently the fastest so far
            is_currently_fastest = fastest_lap_number == lap_number

            if is_currently_fastest:
                return (128, 0, 128, 255)  # Purple - currently fastest so far
            else:
                return (180, 180, 180, 180)  # Light gray - normal or previously fastest
        except Exception as e:
            logger.warning("Error determining sector color: %s", e)
            return (180, 180, 180, 180)

    def get_fastest_sector_lap_number_so_far(
        self, current_lap: int, sector_number: int
    ) -> int:
        """Get the lap number that has the fastest sector time up to and including current_lap"""
        fastest_time = None
        fastest_lap_num = None

        # Look through completed laps up to current lap
        for lap_num in range(1, current_lap + 1):
            if (
                lap_num in self.sector_times
                and len(self.sector_times[lap_num]) >= sector_number
            ):
                sector_time = self.sector_times[lap_num][sector_number - 1]  # 0-indexed
                if fastest_time is None or sector_time < fastest_time:
                    fastest_time = sector_time
                    fastest_lap_num = lap_num

        return fastest_lap_num

    def update_lap_times(self, current_lap_number: int):
        """Update lap times when starting a new lap"""
        try:
            # Skip if we're still on the first lap or going backwards
            if current_lap_number <= 1:
                return

            # Calculate the previous lap time
            previous_lap = current_lap_number - 1
            lap_data = []

            for row in self.rows:
                if int(row["Lap"]) == previous_lap:
                    lap_data.append(row)

            if lap_data:
                # Calculate lap time as difference between last and first frame of the lap
                start_frame = int(lap_data[0]["Record"])
                end_frame = int(lap_data[-1]["Record"])
                lap_time = (end_frame - start_frame) / 25.0  # 25 fps

                # Update laps list
                while len(self.laps) < previous_lap:
                    self.laps.append(0)

                if len(self.laps) >= previous_lap:
                    self.laps[previous_lap - 1] = lap_time
                else:
                    self.laps.append(lap_time)

                # Store in database for real-time color updates
                try:
                    self.db.store_lap(self.session_id, previous_lap, lap_time)

                    # Store sector times if available from real-time tracking
                    if previous_lap in self.real_time_sectors:
                        for sector_num, sector_time in self.real_time_sectors[
                            previous_lap
                        ].items():
                            try:
                                self.db.store_sector(
                                    self.session_id,
                                    previous_lap,
                                    sector_num,
                                    sector_time,
                                )
                            except:
                                pass  # May already exist
                    # Also store any pre-existing sector times from header
                    elif previous_lap in self.sector_times:
                        for sector_num, sector_time in enumerate(
                            self.sector_times[previous_lap], 1
                        ):
                            try:
                                self.db.store_sector(
                                    self.session_id,
                                    previous_lap,
                                    sector_num,
                                    sector_time,
                                )
                            except:
                                pass  # May already exist

                except Exception as e:
                    logger.warning("Error storing lap data in database: %s", e)

        except Exception as e:
            logger.warning("Error updating lap times: %s", e)

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

        # Use MAXIMUM aggressive threading - 3x CPU count for pure image generation workload
        thread_count = max(1, int(cpu_count() * 3.0))
        thread_count = min(thread_count, 36)  # Cap at 36 threads max
        logger.info(f"Using {thread_count} threads for image generation (CPU cores: {cpu_count()})")

        processcount = 0
        while processcount < thread_count:
            logger.info("starting worker %s of %s", processcount + 1, thread_count)
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

    async def generate_images_batched(self, batch_size=600):
        """Generate images in batches with concurrent ffmpeg processing"""
        total_rows = len(self.rows)
        total_batches = (total_rows + batch_size - 1) // batch_size

        # Max concurrent FFmpeg processes for CPU sharing (no semaphore GIL overhead)
        max_concurrent_ffmpeg = 2

        # List to collect all segment paths and ffmpeg tasks
        segment_paths = []
        ffmpeg_tasks = []

        logger.info(
            f"🎬 DELAYED EXECUTION STRATEGY: {total_rows} frames in {total_batches} batches of {batch_size}"
        )
        logger.info(f"📸 Using batched approach with DYNAMIC threading and RESPONSIVE detection")
        logger.info(f"🕐 FFmpeg DELAYED until only 5 image batches remain - MAXIMUM image throughput!")
        logger.info(f"🎞️  Max 2 concurrent FFmpeg processes for CPU sharing (when active)")
        logger.info(f"⚡ Using {int(cpu_count() * 2.0)} image threads (2.0x CPU cores) - NO FFmpeg competition!")
        logger.info("🔧 Smaller batches for more responsive detection and font caching optimization")

        # SYSTEM OPTIMIZATION: Boost Python process priority for better CPU reclaim
        try:
            import os
            current_priority = os.getpriority(os.PRIO_PROCESS, 0)
            os.setpriority(os.PRIO_PROCESS, 0, -5)  # Higher priority than default
            logger.info(f"🚀 Boosted Python process priority from {current_priority} to -5 for better CPU recovery")
        except Exception as e:
            logger.warning(f"Could not adjust process priority: {e}. Consider running with higher privileges.")

        # Queue to track completed image batches ready for ffmpeg
        completed_batches = asyncio.Queue()

        async def image_generator():
            """Generate images for all batches with OVERLAPPING processing"""
            logger.info("🏁 Starting OVERLAPPING background image generation...")

            # Create multiple concurrent image generation tasks
            image_tasks = []

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_rows)
                batch_rows = self.rows[start_idx:end_idx]

                logger.info(
                    f"📸 Starting image generation for batch {batch_idx + 1}/{total_batches} (frames {start_idx}-{end_idx-1})"
                )

                # Create concurrent task for this batch - DON'T WAIT
                image_task = asyncio.create_task(
                    self._generate_and_queue_batch(batch_rows, batch_idx, completed_batches)
                )
                image_tasks.append(image_task)

                # Small stagger to prevent all tasks starting simultaneously
                if batch_idx < total_batches - 1:  # Don't delay after last batch
                    await asyncio.sleep(0.05)  # Very small stagger

            # Wait for all image generation to complete
            await asyncio.gather(*image_tasks)

            # Signal completion
            await completed_batches.put(None)
            logger.info("🏁 All OVERLAPPING image generation completed")

        async def ffmpeg_processor():
            """Process ffmpeg jobs as batches become available - DELAYED START STRATEGY"""
            logger.info("🎬 Starting ffmpeg processor with DELAYED EXECUTION strategy...")
            logger.info(f"🕐 FFmpeg will be DELAYED until only 5 image batches remain (total: {total_batches})")
            processed_batches = 0
            queued_batches = []  # Store batches until we're ready to process them

            try:
                while True:
                    # Wait for next batch or completion signal
                    logger.info("🎬 FFmpeg processor waiting for next batch...")
                    batch_info = await completed_batches.get()

                    if batch_info is None:
                        logger.info("🎬 FFmpeg processor: All image batches completed")

                        # Now process ALL queued batches at once - maximum throughput at the end
                        logger.info(f"🚀 DELAYED EXECUTION: Processing ALL {len(queued_batches)} batches now!")

                        for batch_idx, batch_rows in queued_batches:
                            # Create video segment path
                            segment_path = f"{self.filename}_segment_{batch_idx:04d}.mov"
                            segment_paths.append(segment_path)

                            logger.info(f"🎞️  Starting delayed ffmpeg for batch {batch_idx + 1}...")

                            # Create ffmpeg task
                            ffmpeg_task = asyncio.create_task(
                                self._create_video_segment_with_cleanup(
                                    batch_idx, segment_path, max_concurrent_ffmpeg, batch_rows
                                )
                            )
                            ffmpeg_tasks.append(ffmpeg_task)

                            processed_batches += 1
                            logger.info(f"⚡ Delayed FFmpeg task {processed_batches}/{total_batches} started")

                        break

                    batch_idx, batch_rows = batch_info

                    # Calculate how many image batches are left to generate
                    remaining_image_batches = total_batches - (batch_idx + 1)

                    if remaining_image_batches > 5:
                        # DELAY STRATEGY: Queue this batch for later processing
                        queued_batches.append((batch_idx, batch_rows))
                        logger.info(f"🕐 DELAYING batch {batch_idx + 1} - {remaining_image_batches} image batches still generating (threshold: 5)")
                    else:
                        # START PROCESSING: We're in the final stretch, process immediately
                        if len(queued_batches) > 0:
                            logger.info(f"🚀 THRESHOLD REACHED! Processing {len(queued_batches)} queued batches + current batch")

                            # Process all queued batches first
                            for queued_batch_idx, queued_batch_rows in queued_batches:
                                segment_path = f"{self.filename}_segment_{queued_batch_idx:04d}.mov"
                                segment_paths.append(segment_path)

                                logger.info(f"🎞️  Starting queued ffmpeg for batch {queued_batch_idx + 1}...")

                                ffmpeg_task = asyncio.create_task(
                                    self._create_video_segment_with_cleanup(
                                        queued_batch_idx, segment_path, max_concurrent_ffmpeg, queued_batch_rows
                                    )
                                )
                                ffmpeg_tasks.append(ffmpeg_task)
                                processed_batches += 1

                            queued_batches.clear()  # Clear the queue

                        # Process current batch
                        segment_path = f"{self.filename}_segment_{batch_idx:04d}.mov"
                        segment_paths.append(segment_path)

                        logger.info(f"🎞️  Starting immediate ffmpeg for batch {batch_idx + 1}...")

                        ffmpeg_task = asyncio.create_task(
                            self._create_video_segment_with_cleanup(
                                batch_idx, segment_path, max_concurrent_ffmpeg, batch_rows
                            )
                        )
                        ffmpeg_tasks.append(ffmpeg_task)
                        processed_batches += 1
                        logger.info(f"⚡ Immediate FFmpeg task {processed_batches}/{total_batches} started")

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

        # Wait for both coordination tasks to complete
        logger.info("⏳ Waiting for image generation and ffmpeg coordination to complete...")
        await asyncio.gather(image_task, ffmpeg_task)

        # CRITICAL: Wait for ALL individual ffmpeg tasks to actually finish
        logger.info(f"⏳ Waiting for ALL {len(ffmpeg_tasks)} individual ffmpeg tasks to complete...")
        if ffmpeg_tasks:
            await asyncio.gather(*ffmpeg_tasks)
            logger.info("✅ ALL ffmpeg video segments completed successfully")
        else:
            logger.warning("⚠️  No ffmpeg tasks found - this might indicate an issue")

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
        """Generate images for a specific batch with memory leak prevention"""
        logger.info(
            f"Starting image generation for batch {batch_idx + 1} ({len(batch_rows)} frames)"
        )

        # Use moderate threading to balance performance and memory management
        from concurrent.futures import ThreadPoolExecutor
        import time
        import gc

        # Determine optimal thread count - balance between performance and memory
        max_workers = min(int(cpu_count() * 1.5), 16)  # Conservative threading
        logger.info(f"Using {max_workers} threads for batch {batch_idx + 1}")

        # Pre-create font objects to reduce object creation overhead
        font_cache = {}
        try:
            font_cache['main'] = ImageFont.truetype("font/DejaVuSansCondensed-Bold.ttf", size=self.fontsize)
            font_cache['scale'] = ImageFont.truetype("font/DejaVuSansCondensed-Bold.ttf", size=max(12, self.fontsize // 4))
        except:
            font_cache['main'] = ImageFont.load_default()
            font_cache['scale'] = ImageFont.load_default()

        start_time = time.time()

        # Use ThreadPoolExecutor with memory-safe image generation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for this batch
            futures = [
                executor.submit(self._generate_image_with_cleanup, row, font_cache)
                for row in batch_rows
            ]

            # Wait for completion with progress monitoring
            completed = 0
            for future in futures:
                try:
                    future.result()  # Will raise exception if image generation failed
                    completed += 1

                    # Log progress periodically
                    if completed % 100 == 0 or completed == len(futures):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        logger.info(f"Batch {batch_idx + 1}: {completed}/{len(futures)} images ({rate:.1f}/sec)")

                except Exception as e:
                    logger.error(f"Error generating image in batch {batch_idx + 1}: {e}")
                    raise

        # Force garbage collection after each batch to prevent memory accumulation
        gc.collect()

        total_time = time.time() - start_time
        batch_rate = len(batch_rows) / total_time if total_time > 0 else 0
        logger.info(f"✅ Batch {batch_idx + 1} completed in {total_time:.3f}s ({batch_rate:.1f} img/s)")

    def _generate_image_with_cleanup(self, row, font_cache):
        """Generate an image with proper memory cleanup to prevent leaks"""
        try:
            # Generate the image using the existing method
            self.generate_image(row, font_cache=font_cache)

        except Exception as e:
            # Log error but don't fail the whole batch
            logger.error(f"Error generating image for frame {row.get('Record', 'unknown')}: {e}")
            raise

    async def _generate_and_queue_batch(self, batch_rows, batch_idx, completed_batches):
        """Generate images for a batch and queue when ready - with performance profiling"""
        def profile_threads():
            """Helper function to profile thread and process status"""
            active_threads = threading.active_count()
            thread_names = [t.name for t in threading.enumerate()]
            logger.info(f"🔍 PROFILING BATCH {batch_idx + 1}: {active_threads} active threads: {thread_names}")

            # Monitor CPU usage per process
            try:
                import psutil
                current_process = psutil.Process()
                cpu_percent = current_process.cpu_percent(interval=0.1)
                memory_mb = current_process.memory_info().rss / 1024 / 1024
                logger.info(f"🔍 PROCESS STATS: CPU={cpu_percent:.1f}%, Memory={memory_mb:.1f}MB")

                # Count threads in this Python process
                thread_count = current_process.num_threads()
                logger.info(f"🔍 PYTHON THREADS: {thread_count} threads in main process")

                # Check for any lingering asyncio activity
                try:
                    loop = asyncio.get_running_loop()
                    task_count = len([t for t in asyncio.all_tasks(loop) if not t.done()])
                    logger.info(f"🔍 ASYNCIO TASKS: {task_count} active tasks")
                except:
                    logger.info(f"🔍 ASYNCIO: No running loop detected")

            except Exception as e:
                logger.warning(f"Could not get process stats: {e}")

        # Profile BEFORE processing
        logger.info(f"🔍 === PROFILING BEFORE BATCH {batch_idx + 1} ===")
        profile_threads()

        # DYNAMIC THREADING: Scale based on actual FFmpeg process count in real-time
        # This solves the core issue - detecting when FFmpeg completes and scaling back up

        try:
            import psutil
            # Count active FFmpeg processes on the system
            ffmpeg_processes = [p for p in psutil.process_iter(['name']) if 'ffmpeg' in p.info['name'].lower()]
            active_ffmpeg_count = len(ffmpeg_processes)

            # AGGRESSIVE SCHEDULER COMBAT: Scale threads based on FFmpeg processes
            # cpus*2.0 when no FFmpeg, cpus*0.5 when FFmpeg is running

            if active_ffmpeg_count == 0:
                # NO FFMPEG RUNNING: Apply adaptive scaling
                max_workers = int(cpu_count() * 2.0)
                max_workers = max(1, min(max_workers, 32))  # Higher cap for aggressive scaling
                logger.info(f"🚀 BATCH {batch_idx + 1}: NO FFMPEG - AGGRESSIVE SCALING - Using {max_workers} threads (2.0x CPU)")

                # MAXIMUM EFFICIENCY: Skip pauses when thermal isn't the issue
                # Maintain consistent high performance without interruption
                logger.info(f"� Maintaining peak performance for batch {batch_idx + 1} (no thermal constraints)")

                # SYSTEM OPTIMIZATION: Process warm-up to combat scheduler inertia
                if batch_idx > 0:  # Only after first batch (when FFmpeg might have been running)
                    logger.info(f"🔥 Warming up process pool to combat system scheduler inertia...")
            else:
                # FFMPEG RUNNING: Conservative scaling to share CPU resources
                max_workers = int(cpu_count() * 0.5)
                max_workers = max(1, min(max_workers, 8))  # Conservative cap for shared CPU
                logger.info(f"⚖️  BATCH {batch_idx + 1}: {active_ffmpeg_count} FFMPEG ACTIVE - SHARED CPU - Using {max_workers} threads (0.5x CPU)")

        except Exception as e:
            # Fallback: Use aggressive scaling if we can't detect FFmpeg
            logger.warning(f"Could not detect FFmpeg processes: {e}")
            max_workers = int(cpu_count() * 2.0)  # Default to aggressive
            max_workers = max(1, min(max_workers, 32))
            logger.info(f"🔄 BATCH {batch_idx + 1}: FALLBACK AGGRESSIVE - Using {max_workers} threads (2.0x CPU)")
            logger.info(f"⚠️  Assuming no FFmpeg running due to detection failure")

        # Pre-create font objects to reduce GIL contention during image generation
        font_cache = {}
        try:
            font_cache['main'] = ImageFont.truetype("font/DejaVuSansCondensed-Bold.ttf", size=self.fontsize)
            font_cache['scale'] = ImageFont.truetype("font/DejaVuSansCondensed-Bold.ttf", size=max(12, self.fontsize // 4))
        except:
            font_cache['main'] = ImageFont.load_default()
            font_cache['scale'] = ImageFont.load_default()

        # Memory optimization: Pre-allocate reusable image canvas to reduce memory overhead
        # Note: With processes, each worker will create its own copy, but this reduces initial allocation
        image_template = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        # PROFILING: Monitor thread creation
        logger.info(f"🔍 CREATING ThreadPoolExecutor with {max_workers} workers...")
        start_time = time.time()

        # Use ThreadPoolExecutor for FULL feature access with dynamic scaling
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Profile AFTER thread pool creation
            logger.info(f"🔍 === PROFILING AFTER ThreadPoolExecutor CREATION ===")
            profile_threads()

            # AGGRESSIVE THREAD OPTIMIZATION: Combat scheduler inertia for sustained performance
            if batch_idx > 0 and active_ffmpeg_count == 0:
                # AGGRESSIVE: Thread warm-up to ensure OS scheduler prioritizes Python threads
                warmup_start = time.time()
                # Double warm-up for batches 4+ since that's where we see the cliff
                warmup_iterations = 2 if batch_idx >= 3 else 1
                for iteration in range(warmup_iterations):
                    warmup_futures = [executor.submit(lambda: None) for _ in range(max_workers)]
                    for future in warmup_futures:
                        future.result()
                warmup_time = time.time() - warmup_start
                logger.info(f"🔥 Thread pool SUPER-AGGRESSIVELY warmed up with {max_workers} threads x{warmup_iterations} iterations in {warmup_time:.3f}s")

                # Profile AFTER warmup
                logger.info(f"🔍 === PROFILING AFTER WARMUP ===")
                profile_threads()

            # PROFILING: Monitor image generation start
            image_start_time = time.time()
            logger.info(f"🔍 STARTING image generation for {len(batch_rows)} frames...")

            # Submit all batch rows for processing with FULL dashboard generation
            # Use the complete generate_image method for all features
            futures = [
                executor.submit(
                    self._generate_image_optimized,
                    row,
                    font_cache,
                    image_template
                )
                for row in batch_rows
            ]

            # Profile AFTER submitting all futures
            logger.info(f"🔍 === PROFILING AFTER SUBMITTING {len(futures)} FUTURES ===")
            profile_threads()

            # Wait for all images in this batch to complete with progress monitoring
            completed = 0
            for i, future in enumerate(futures):
                future.result()  # This will raise an exception if image generation failed
                completed += 1

                # Log progress and profile every 100 images
                if completed % 100 == 0 or completed == len(futures):
                    elapsed = time.time() - image_start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(f"🔍 PROGRESS: {completed}/{len(futures)} images ({rate:.1f} images/sec)")

                    # Quick profile during processing to catch GIL issues
                    if completed % 200 == 0:
                        logger.info(f"🔍 === MID-PROCESSING PROFILE ({completed} images) ===")
                        profile_threads()

        # Profile AFTER thread pool cleanup
        logger.info(f"🔍 === PROFILING AFTER ThreadPoolExecutor CLEANUP ===")
        profile_threads()

        # Explicit garbage collection after batch completion to reduce memory pressure
        import gc
        gc.collect()

        total_time = time.time() - start_time
        batch_rate = len(batch_rows) / total_time if total_time > 0 else 0

        # PERFORMANCE TRACKING: Record batch performance for analysis
        self.batch_performance_history.append({
            'batch': batch_idx + 1,
            'rate': batch_rate,
            'time': total_time,
            'threads': max_workers
        })

        # ADAPTIVE PERFORMANCE ANALYSIS: Detect scheduler cliff in real-time
        if len(self.batch_performance_history) > 1:
            prev_rate = self.batch_performance_history[-2]['rate']
            rate_change = (batch_rate - prev_rate) / prev_rate * 100 if prev_rate > 0 else 0

            if rate_change < -20:  # More than 20% performance drop
                logger.warning(f"🚨 PERFORMANCE CLIFF DETECTED: Batch {batch_idx + 1} rate {batch_rate:.1f} img/s ({rate_change:+.1f}% vs previous)")
            elif rate_change > 5:  # Performance improvement
                logger.info(f"📈 PERFORMANCE GAIN: Batch {batch_idx + 1} rate {batch_rate:.1f} img/s ({rate_change:+.1f}% vs previous)")
            else:
                logger.info(f"📊 PERFORMANCE STABLE: Batch {batch_idx + 1} rate {batch_rate:.1f} img/s ({rate_change:+.1f}% vs previous)")

        logger.info(f"✅ BATCH {batch_idx + 1} IMAGES COMPLETED in {total_time:.3f}s ({batch_rate:.1f} img/s) - ready for ffmpeg")

        # Final profiling
        logger.info(f"🔍 === FINAL PROFILING BATCH {batch_idx + 1} ===")
        profile_threads()

    def _generate_image_optimized(self, row, font_cache, image_template=None):
        """Optimized image generation with reduced GIL contention and memory reuse"""
        try:
            # Use the existing generate_image method but with pre-cached fonts and image template
            # This reduces object creation overhead in threads
            return self.generate_image(row, font_cache=font_cache, image_template=image_template)
        except Exception as e:
            logger.error(f"Error generating image for frame {row.get('Record', 'unknown')}: {e}")
            raise

    async def _generate_and_queue_batch(self, batch_rows, batch_idx, completed_batches):
        """Generate images for a batch and queue when ready"""
        logger.info(
            f"🔄 OVERLAPPING: Starting batch {batch_idx + 1} ({len(batch_rows)} frames)"
        )

        # Generate images for this batch
        await self._generate_batch_images(batch_rows, batch_idx)

        # Queue this batch for FFmpeg processing
        await completed_batches.put((batch_idx, batch_rows))
        logger.info(
            f"✅ OVERLAPPING: Batch {batch_idx + 1} images ready - queued for ffmpeg"
        )

    async def _create_video_segment(
        self, batch_idx, segment_path, max_concurrent, batch_rows
    ):
        """Create a video segment from a batch of images without semaphore GIL overhead"""
        logger.info(f"Starting ffmpeg for batch {batch_idx + 1}...")

        # Simple process limiting without asyncio semaphores (eliminates GIL contention)
        import time
        while True:
            # Count active FFmpeg processes using system tools (no Python GIL involved)
            try:
                import subprocess
                result = subprocess.run(['pgrep', '-c', 'ffmpeg'], capture_output=True, text=True)
                active_count = int(result.stdout.strip()) if result.returncode == 0 else 0

                if active_count < max_concurrent:
                    break

                logger.info(f"Waiting for FFmpeg slot (current: {active_count}/{max_concurrent})...")
                time.sleep(0.1)  # Brief wait without blocking other tasks
            except:
                # Fallback: just proceed if we can't count processes
                break

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

            # Run FFmpeg directly without asyncio thread pool to eliminate GIL contention
            self._run_ffmpeg_with_nice(cmd)
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

    def _run_ffmpeg_with_nice(self, cmd):
        """Run FFmpeg command with lower priority as a true subprocess, no GIL interference"""
        import subprocess
        import os

        # Get the compiled command
        ffmpeg_cmd = cmd.compile()

        # Run with nice priority (+15 = much lower priority, allows Python to preempt easily)
        nice_cmd = ['nice', '-n', '15'] + ffmpeg_cmd

        try:
            # Run the nice command directly as a subprocess - no thread pool, no GIL
            result = subprocess.run(nice_cmd, check=True, capture_output=True, text=True)
            logger.info(f"FFmpeg completed with nice priority (true subprocess)")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr}")
            raise
        except Exception as e:
            logger.warning(f"Nice command failed, falling back to normal priority: {e}")
            # Fallback to original command if nice fails
            return cmd.run()

    def _run_ffmpeg_background(self, cmd, segment_path, batch_idx):
        """Start FFmpeg as a background subprocess without asyncio/thread overhead"""
        import subprocess
        import os

        # Get the compiled command
        ffmpeg_cmd = cmd.compile()
        nice_cmd = ['nice', '-n', '15'] + ffmpeg_cmd

        try:
            # Start subprocess without waiting - true background process
            process = subprocess.Popen(
                nice_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"🎞️  FFmpeg process {batch_idx + 1} started as PID {process.pid} (background)")
            return process
        except Exception as e:
            logger.error(f"Failed to start FFmpeg background process: {e}")
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
        font_scale=1.0,
    ):
        if font is None:
            # Apply font scaling
            scaled_fontsize = int(self.fontsize * font_scale)
            try:
                font = ImageFont.truetype(
                    "font/DejaVuSansCondensed-Bold.ttf", size=scaled_fontsize
                )
            except Exception:
                font = ImageFont.load_default()
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

    def generate_image(self, row=dict, font_cache=None, image_template=None):
        # Initialize or use provided font cache for performance
        if font_cache is None:
            font_cache = {}
            try:
                font_cache['main'] = ImageFont.truetype("font/DejaVuSansCondensed-Bold.ttf", size=self.fontsize)
                font_cache['scale'] = ImageFont.truetype("font/DejaVuSansCondensed-Bold.ttf", size=max(12, self.fontsize // 4))
            except:
                font_cache['main'] = ImageFont.load_default()
                font_cache['scale'] = ImageFont.load_default()

        # Use image template if provided, otherwise create new image
        if image_template is not None:
            img = image_template.copy()  # Reuse pre-allocated canvas
        else:
            img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

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

        draw = ImageDraw.Draw(img)

        # Current lap info and timer
        current_lap_number = int(row["Lap"])
        current_lap_time = 0.0

        # Calculate positioning for lap info
        # Align with speed indicator and graph margin (left side)
        lap_info_x = self.width * 0.08  # Same as speed indicator
        # Align top with trackmap (which is at y=50)
        lap_info_y = 50  # Same as trackmap top position

        if current_lap_number > 0:
            # Check if we're on a new lap and should update lap times
            if current_lap_number != self.last_lap_number:
                self.update_lap_times(current_lap_number)
                self.last_lap_number = current_lap_number

            # Calculate current lap time
            current_lap_time = self.get_current_lap_time(row)

            # Check if current lap just completed (progress = 100%) and store immediately
            current_frame = int(row["Record"])
            lap_frames = []
            for r in self.rows:
                if int(r["Lap"]) == current_lap_number:
                    lap_frames.append(int(r["Record"]))

            if lap_frames:
                lap_start = min(lap_frames)
                lap_end = max(lap_frames)
                if lap_end > lap_start:
                    current_progress = (current_frame - lap_start) / (
                        lap_end - lap_start
                    )
                    # If lap just completed, store it immediately for real-time coloring
                    if (
                        current_progress >= 0.99
                        and current_lap_number not in self.completed_sectors
                    ):
                        try:
                            self.db.store_lap(
                                self.session_id, current_lap_number, current_lap_time
                            )
                            # Mark as completed to avoid duplicate storage
                            self.completed_sectors[current_lap_number] = set([1, 2, 3])
                        except:
                            pass  # May already exist

            # Display current lap number and timer on same line
            if current_lap_time > 0:
                minutes, seconds = self.convert_seconds(int(current_lap_time))
                timer_text = f" - {minutes}:{str(int(seconds)).zfill(2)}.{str(int((current_lap_time % 1) * 100)).zfill(2)}"
                lap_with_timer = lap + timer_text
            else:
                lap_with_timer = lap

            self.generate_textbox(
                draw=draw,
                x=lap_info_x,
                y=lap_info_y,
                text=lap_with_timer,
                align="left",
                color=(255, 255, 255, 200),  # White color like other text
            )

            # Display sector counter on next line with individual sector colors
            self.draw_sector_counter_with_colors(
                draw, row, lap_info_x, lap_info_y + self.fontsize * 1.2
            )  # Show lap times after completing at least one lap (keep them visible even in lap 0)
        if current_lap_number > 1 or self.hasShownLaps:
            lapcnt = 1
            self.hasShownLaps = True
            current_lap_for_display = current_lap_number
            if current_lap_for_display == 0:
                # make sure we show laptimes for the in-lap
                current_lap_for_display = 999
            for lap in self.laps:
                if lapcnt < current_lap_for_display:
                    # Determine color for this lap time using current lap as context
                    # This ensures previously fastest laps turn gray when beaten
                    lap_color = self.get_lap_color(lapcnt, lap, current_lap_number)

                    # Use 80% font for previous laps
                    font_scale = 0.8

                    self.generate_textbox(
                        draw=draw,
                        x=lap_info_x,
                        y=lap_info_y
                        + self.fontsize
                        * 2.5  # Offset for current lap and sector counter
                        + self.fontsize * 0.15 * lapcnt
                        + self.fontsize * lapcnt * font_scale,
                        # Prepare lap time formatting with zero-padded seconds (no sector times)
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
                        color=lap_color,
                        font_scale=font_scale,
                    )
                lapcnt = lapcnt + 1

        self.generate_textbox(
            draw=draw,
            x=self.width * 0.08,
            y=self.height * 0.8,
            text=speed,
            font=font_cache.get('main'),
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
            draw=draw, x=self.width * 0.832, y=self.height * 0.74, text=lean, font=font_cache.get('main')
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

        # CRITICAL: Close all images to prevent memory leaks
        try:
            # Close the track position image that was created by draw_position
            if drawimage:
                drawimage.close()
            # Close the main image
            if img:
                img.close()
        except Exception as e:
            logger.warning(f"Error closing images: {e}")
            # Don't raise here - image is already saved
