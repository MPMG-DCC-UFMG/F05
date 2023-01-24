import shutil #salvar arq no disco
from typing import List #lista de imagens
import os
from fastapi import FastAPI, UploadFile, File

from fastapi.responses import HTMLResponse #pagina

app = FastAPI()
#receber id img
#retornar csv final

@app.post("/obras/")
async def upload_images(files: List[UploadFile] = File(...),id=0):
    print(os.listdir())
    os.chdir("obras")
    
    #criar a pasta api/img/
    mypath = 'api/img_obras_'+str(id)+'/'
    if not os.path.isdir(mypath):
        os.mkdir(mypath)

    #salvar imagens na pasta "img/"
    for file in files:
        with open(mypath + file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer) 

    #rodar o modelo nas imgs baixadas
    os.system("python3 Framework.py --path " + mypath + "  --single_folder True")
    #retornar arquivo csv
    f = open(mypath + "predictions.csv","r")
    response = f.read()
    f1 = open(mypath + "predictions_final.csv","r")
    response1 = f1.read()

    #remover pasta com dados anteriores
    shutil.rmtree(mypath)

    return {response1},{response}

@app.post("/pavimentacao/")
async def upload_images(files: List[UploadFile] = File(...),id=0):
    #criar a pasta api/img/
    print(os.listdir())
    os.chdir("pavimentacao")
    
    mypath = 'api/img_pavimentacao_' + str(id) + '/'
    
    if not os.path.isdir(mypath):
        os.mkdir(mypath)

    for file in files:
        #salvar imagens na pasta "img/"
        with open(mypath + file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer) 

    #rodar o modelo nas imgs baixadas
    os.system("python3 mpmg_prediction.py --path " + mypath)
    f = open("classification_list.csv","r")
    response = f.read()
    
    #remover pasta com dados anteriores
    shutil.rmtree(mypath)
    return {response}

#chama o post
@app.get("/")
async def main():
    #os h1 são provisorios
    content = """
<body>
<h2>.</h2>
<h1>obras</h1>
<form action="/obras/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
</form>
<h1>pavimentação</h1>
<form action="/pavimentacao/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
