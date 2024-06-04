import asyncio
import codecs
import csv
import io
import os
import tempfile
import time

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

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
    await output.generate_images()
    outputfile_path = await output.generate_movie()

    return FileResponse(
        path=outputfile_path, filename=outputfile_path, media_type="text/mp4"
    )


@app.get("/")
async def root(request: Request):
    content = """
<body>
Racebox CSV file:
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>

</body>
    """
    return HTMLResponse(content=content)
