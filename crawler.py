import pycurl
import scrapy
import csv
import os

with open('general_data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(["ID", "Título","Situação","Município - UF","Localização","CEP","Endereço","Bairro","Termo/Convênio","Fim da Vigência","Situação do Termo","Data da Última Vistoria","Percentual de Execução"])

if not os.path.exists("IMAGES"):
    os.mkdir("IMAGES")

class SIMECSpider(scrapy.Spider):
    name = "simec_spider"
    start_urls = [f'http://simec.mec.gov.br/painelObras/vistoria.php?obra={i}' for i in range(8062, 31979)]

    def parse(self, response):
        id = response.url.split("=")[1]
        data = [
            id, #ID
            response.xpath(".//h2/text()").get(), #Titulo
            response.xpath(".//dd/span/text()").getall()[0][60:], #Situacao
            response.xpath(".//dd/text()").getall()[0], #Municipio - UF
            response.xpath(".//dd/a").css("::attr(coord)").get(), #Coordenadas
            response.xpath(".//dd/text()").getall()[1], #CEP
            response.xpath(".//dd/text()").getall()[2], #Endereco
            response.xpath(".//dd/text()").getall()[3], #Bairro
            response.xpath(".//dd/text()").getall()[4], #Termo/Convenio
            response.xpath(".//dd/text()").getall()[5], #Fim da Vigencia
            response.xpath(".//dd/text()").getall()[6], #Situacao do Termo
            response.xpath(".//dd/text()").getall()[7], #Data da Ultima Vistoria
            response.xpath(".//dd/text()").getall()[9] #Percentual de Execucao
        ]
        images = response.xpath(".//a").css(".img_foto::attr(src)").getall()

        with open('general_data.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(data)

        if not os.path.exists("IMAGES/" + id):
            os.mkdir("IMAGES/" + id)

        for i, img in enumerate(images):
            file = open(f"IMAGES/{id}/{i}.jpeg", "wb")
            crl = pycurl.Curl()
            crl.setopt(crl.URL, img)
            crl.setopt(crl.WRITEDATA, file)
            crl.perform()
            crl.close()
            file.close()
