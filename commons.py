import logging
import os
import sys
from datetime import datetime, timedelta

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
    def __init__(self, rows):
        # initialize global vars
        self.maxspeed = float(0)
        self.maxleanleft = float(0)
        self.maxleanright = float(0)
        self.maxaccg = float(0)
        self.maxdecg = float(0)
        # iterate through rows,
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

    def generate_image(self, row):
        font = ImageFont.truetype("font/OpenSans-Bold.ttf", size=200)

        img = Image.new("RGBA", (3840, 2180))

        draw = ImageDraw.Draw(img)
        draw_point = (0, 0)
        draw.multiline_text(draw_point, row["Record"], font=font, fill=(0, 0, 0))

        draw_point = (0, 200)

        draw.multiline_text(draw_point, str(row), font=font, fill=(0, 0, 0))

        img.save("Image.png")
