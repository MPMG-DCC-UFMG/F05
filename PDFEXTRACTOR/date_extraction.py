import sys
import os
import pytesseract
import pdfplumber
from pdf2image import convert_from_path
import nltk
from unidecode import unidecode
from difflib import SequenceMatcher


class DocumentExtractor():
    def __init__(self, path):
        self.dir_path = path

    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def get_text_from_pdf(self):
        pages = convert_from_path(self.path, 300, fmt='jpeg')
        extracted_text = [pytesseract.image_to_string(img, lang='por') for img in pages]
        return (extracted_text)

    def read_file(self, file_path):
        pdf = pdfplumber.open(file_path)
        text = []
        for page in pdf.pages:
            text.append(page.extract_text())
        if None in text:
            print("Handling file as image...")
            text = self.get_text_from_pdf()
        text = ' (NEWPAGE) '.join(text)
        return text

    def clean_text(self, text, remove_stop_words=True):
        stop_words = nltk.corpus.stopwords.words("portuguese")
        text_by_words = text.replace('\n', ' ').split(' ')
        if remove_stop_words:
            formated_text = [unidecode(word.lower()) for word in text_by_words if (len(word) >= 1) & (word.lower() not in stop_words)]
        else:
            formated_text = [unidecode(word.lower()) for word in text_by_words if (len(word) >= 1)]
        return formated_text

    def get_prazo_clauses(self, text):
        sentences = []
        for idx, i in enumerate(text):
            if self.similar('clausula', i) > 0.7:
                clause = text[idx:]
                if any(test in clause[:10] for test in ['prazo', 'vigencia']):
                    for j in range(1, len(clause)):
                        if self.similar('clausula', clause[j]) > 0.7:
                            clause = clause[:j]
                            break
                    print(' '.join(clause))
                    sentences.append(' '.join(clause))
        return sentences

    def read_all_documents(self):
        prazos = []
        if os.path.isdir(self.dir_path):
            for file in sorted(os.listdir(self.dir_path)):
                print('Reading File {}'.format(file))
                self.path = '{}/{}'.format(self.dir_path, file)
                text = self.read_file(self.path)
                print('Cleaning Texts')
                text = self.clean_text(text, remove_stop_words=False)
                print('Getting Prazos')
                sentences = self.get_prazo_clauses(text)
                prazos.append(sentences)
                print('\n')
        else:
            print('Reading File {}'.format(self.dir_path))
            self.path = self.dir_path
            text = self.read_file(self.path)
            print('Cleaning Texts')
            text = self.clean_text(text, remove_stop_words=False)
            print('Getting Prazos')
            sentences = self.get_prazo_clauses(text)
            prazos.append(sentences)
            print('\n')
        return prazos

    def write_prazos(self, prazos):
        erros = []
        with open('output.txt', 'w') as f:
            for i in range(len(prazos)):
                f.write('FILE: {}\n'.format(i))
                if len(prazos) == 0:
                    erros.append(i)
                    f.write('NONE FOUND! \n')
                for _string in prazos[i]:
                    f.write(str(_string) + '\n')
                f.write('\n')


if __name__ == "__main__":
    path = sys.argv[1]
    # path = './documents/contratante'
    extractor = DocumentExtractor(path)
    prazos = extractor.read_all_documents()
    extractor.write_prazos(prazos)
