import asyncio
import codecs
import csv
import io
import os
import sys
import tempfile

from commons import dashGenerator, setup_logging

logger = setup_logging()


async def parse_file(filename):
    tmpfile = tempfile.SpooledTemporaryFile(mode="r+")

    uploaddata = []
    uploadHeader = []
    beyondHeader = False
    f = open(filename)
    for line in f:
        if beyondHeader == True:
            tmpfile.write(line)
        else:
            uploadHeader.append(line)
        if line == "\n":
            beyondHeader = True
            logger.info("found blank line")

    tmpfile.seek(0)

    csvReader = csv.DictReader(tmpfile)
    uploaddata = []

    for row in csvReader:
        uploaddata.append(row)

    f.close()
    csvReader = None

    output = dashGenerator(
        rows=uploaddata, header=uploadHeader, width=1920, filename=filename
    )
    await output.generate_images()
    outputfile_path = await output.generate_movie()

    return ("output file: %s", outputfile_path)


asyncio.run(parse_file(sys.argv[1]))
