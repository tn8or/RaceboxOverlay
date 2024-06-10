import asyncio
import hashlib
import json
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
from collections import deque
from datetime import datetime, timedelta

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

        self.tmpdir = tempfile.TemporaryDirectory()
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
        self.foldername = (
            self.tmpdir.name
            + hashlib.md5(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
            + "/"
        )
        self.log = []

        if width == 1920:
            self.fontsize = 50
            self.height = 1080
            self.width = width
            self.maxtracksize = 500
            self.polygonwidth = 10
        if width == "uhd":
            self.fontsize = 100
            self.width = width
            self.height = 2160
            self.maxtracksize = 1000
            self.polygonwidth = 20

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

        logger.info("general stats gathered")

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

        self.latscaling = int(self.trackwidth * 0.95) / self.lat
        self.lonscaling = int(self.trackheight * 0.95) / self.lon

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

        lat = int((lat - self.correctedminlat) * self.latscaling / 10000000)
        lon = int((lon - self.correctedminlon) * self.lonscaling / 10000000)

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

        global trackimage
        trackimage = self.generate_trackimage(polygon)

    def generate_trackimage(self, polygon):
        trackimage = Image.new(
            "RGBA",
            (int(self.trackwidth), int(self.trackheight)),
            (255, 255, 255, 0),
        )

        draw = ImageDraw.Draw(trackimage)
        # draw.polygon(polygon, width=int(self.polygonwidth * 1.5), outline=(150, 150, 150, 75))
        draw.polygon(polygon, width=self.polygonwidth, outline=(0, 0, 0))
        return trackimage

    def thread_worker(self, q):
        while True:
            row = q.get()
            self.generate_image(row=row)
            q.task_done()

    async def generate_images(self):
        q = queue.Queue()
        # iterate through rows again, generate one image per row

        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()
        threading.Thread(target=self.thread_worker, args=[q], daemon=True).start()

        logger.info("all threads started")
        self.log.append("all threads started")

        for row in self.rows:
            q.put(row)

        logger.info("all rows added to queue")
        self.log.append("all rows added to queue")

        q.join()
        logger.info("All images built")

    async def generate_movie(self):

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
        font="font/OpenSans-Bold.ttf",
        color=(255, 255, 255, 200),
        align="center",
    ):
        font = ImageFont.truetype(font, size=self.fontsize)
        draw_point = (x, y)
        length = draw.textlength(text, font=font)
        draw.rounded_rectangle(
            xy=(
                (x, y + self.fontsize * 0.2),
                (x + length, y + self.fontsize + self.fontsize * 0.2),
            ),
            radius=5,
            fill=(50, 50, 50, 50),
            outline=None,
        )
        draw.text(draw_point, text, font=font, fill=color, align=align)

    def generate_image(self, row=dict):
        frame = row["Record"]
        speed = row["Speed"]
        speed = speed[:-3] + " km/h"
        lap = "Lap: " + row["Lap"]
        lean = float(row["LeanAngle"])
        if lean > 0:
            lean = str(round(lean))
        else:
            lean = str(round(-lean))

        lean = lean + "°"

        date = row["Time"].split("T")
        time = date[1].split(".")

        filename = self.foldername + "frame" + frame.rjust(8, "0") + ".png"

        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        draw = ImageDraw.Draw(img)

        # left top"
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

        # screen bottom
        self.generate_textbox(
            draw=draw, x=self.width * 0.2, y=self.height * 0.9, text=lap, align="left"
        )

        self.generate_textbox(
            draw=draw,
            x=self.width * 0.4,
            y=self.height * 0.9,
            text=speed,
        )

        self.generate_textbox(
            draw=draw, x=self.width * 0.8, y=self.height * 0.9, text=lean
        )

        img.paste(
            trackimage,
            (self.width - int(self.trackwidth) - int(self.trackwidth * 0.1), 50),
        )

        img.save(filename, "PNG")
        # logger.info("Saved file " + filename)
        self.log.append("Saved file " + filename)
