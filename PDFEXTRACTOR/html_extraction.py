import sys
import datetime
import codecs
import os
import bs4 as bs
from dateutil.relativedelta import relativedelta
import glob


class HTMLExtractor():
    def __init__(self, path):
        self.path = path

    def path_extraction(self):
        files = sorted(glob.glob(os.path.join(self.path, '*.html')))

        noinfo_count = 0
        sign_dates, end_dates = [], []

        for file in files:
            a = self.extract_dates(file)
            if a is not None:
                sign_dates.append(a[0])
                end_dates.append(a[1])
            else:
                noinfo_count += 1
        return (sign_dates, end_dates)

    def extract_dates(self, html_path):
        file = codecs.open(html_path, 'r').read()

        id_signature = 'ctl00_ContentPlaceHolderConteudo_dataAssinaturaLabel'
        id_prazo = 'ctl00_ContentPlaceHolderConteudo_prazoTotalDiretaLabel'

        soup = bs.BeautifulSoup(file, 'lxml')

        supa = soup.find('span', attrs={'id': id_signature})
        signature_date = supa.contents

        if len(signature_date) > 0:

            supa = soup.find('span', attrs={'id': id_prazo})
            prazo = supa.contents

            if (int(prazo[0]) % 30) == 0:
                date_1 = datetime.datetime.strptime(signature_date[0], "%d/%m/%Y")
                end_date = date_1 + relativedelta(months=+int(prazo[0])/30)
            else:
                date_1 = datetime.datetime.strptime(signature_date[0], "%d/%m/%Y")
                end_date = date_1 + datetime.timedelta(days=int(prazo[0]))

            with open('html_output.txt', 'a') as f:
                f.write('{}:\n'.format(html_path))
                f.write(signature_date[0] + '\n')
                f.write(end_date.strftime('%d/%m/%Y') + '\n')
                f.write('\n')

            return (signature_date[0], end_date.strftime('%d/%m/%Y'))
        else:
            with open('html_output.txt', 'a') as f:
                f.write('{}:\n'.format(html_path))
                f.write('NO DATA AVAILABLE\n')
                f.write('\n')
            return None


if __name__ == "__main__":
    os.system('rm html_output.txt')
    path = sys.argv[1]
    if os.path.isdir(path):
        extractor = HTMLExtractor(path)
        a = extractor.path_extraction()
    else:
        extractor = HTMLExtractor(path)
        a = extractor.extract_dates(html_path=path)
