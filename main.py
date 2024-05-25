import asyncio
import codecs
import csv

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse

from commons import dashGenerator, setup_logging

# from fastapi.templating import Jinja2Templates


# from typing import Annotated, Union


logger = setup_logging()
app = FastAPI()


@app.post("/uploadfiles/")
async def create_upload_file(file: UploadFile):
    csvReader = csv.DictReader(codecs.iterdecode(file.file, "utf-8"))
    data = []

    for row in csvReader:
        data.append(row)

    output = dashGenerator(data)

    file.file.close()

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
