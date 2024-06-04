import asyncio
import codecs
import csv
import io
import os
import tempfile
import time

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse

from commons import dashGenerator, setup_logging

# from fastapi.templating import Jinja2Templates


# from typing import Annotated, Union


logger = setup_logging()
app = FastAPI()


@app.post("/uploadfiles/")
async def create_upload_file(file: UploadFile):

    tmpfile = tempfile.SpooledTemporaryFile(mode="r+")

    uploaddata = []
    uploadHeader = []
    beyondHeader = False
    with file.file as f:
        for line in io.TextIOWrapper(file.file, encoding="utf-8"):
            #            logger.info(line)
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

    file.file.close()
    csvReader = None

    output = dashGenerator(
        rows=uploaddata, header=uploadHeader, width=1920, filename=file.filename
    )
    for function in output.generate_images, output.generate_movie:
        t1 = time.perf_counter(), time.process_time()
        await function()
        t2 = time.perf_counter(), time.process_time()
        logger.info(f"{function.__name__}()")
        logger.info(f" Real time: {t2[0] - t1[0]:.2f} seconds")
        logger.info(f" CPU time: {t2[1] - t1[1]:.2f} seconds")

    return {"filename": file.filename, "size": file.size, "output": output}


@app.get("/")
async def root(request: Request):
    content = """
<body>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
