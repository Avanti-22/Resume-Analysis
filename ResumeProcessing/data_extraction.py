# importing all the libraries
import io
import os
import re
import nltk
import pandas as pd

# import PyPDF2
# for pdf to txt
import pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from io import StringIO
from pdfminer.pdfpage import PDFPage

import docx2txt
import constants as cs
import string
import utils
import pprint
from spacy.matcher import matcher
import multiprocessing as mp
import warnings

from constants import STOPWORDS
warnings.filterwarnings('ignore')

def add_file():
    input("Upload your resume")
resume=add_file()

# Extract txt from pdf
# pdf_path="code\Resume.pdf"
def pdf_to_txt(pdf_path):
    resource_manager = PDFResourceManager(caching=True)
    
    # create a string object that will contain the final text the representation of the pdf. 
    out_text = StringIO()
    codec = 'utf-8'
    laParams = LAParams()
    
    # Create a TextConverter Object:
    text_converter = TextConverter(resource_manager, out_text, laparams=laParams)
    fp = open(pdf_path, 'rb')
    #Create a PDF interpreter object

    interpreter = PDFPageInterpreter(resource_manager, text_converter)
    
    # We are going to process the content of each page of the original PDF File    
    for page in PDFPage.get_pages(fp, pagenos=set(), maxpages=0, password="", caching=True, check_extractable=True):
        interpreter.process_page(page)

    
    # Retrieve the entire contents of the “file” at any time 
    text = out_text.getvalue()

    # Closing all the ressources we previously opened

    fp.close()
    text_converter.close()
    out_text.close()
    
    return text
# retext = pdf_to_txt(pdf_path)
# print(pdf_to_txt(pdf_path))

# Extract txt from
# docx_path="Resume.docx"
def docx_to_txt(docx_path):
    temp = docx2txt.process(docx_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)
# print(docx_to_txt(docx_path))

# # detect file extension and call above functions accordingly
# def extract_text(file_path, extension):
#     '''
#     Wrapper function to detect the file extension and call text extraction function accordingly

#     :param file_path: path of file of which text is to be extracted
#     :param extension: extension of file `file_name`
#     '''
#     text = ''
#     if extension == '.pdf':
#         for page in pdf_to_txt(file_path):
#             text += ' ' + page
#     elif extension == '.docx' or extension == '.doc':
#         text = docx_to_txt(file_path)
#     return text


# def extract_entity_sections(text):
#     '''
#     Helper function to extract all the raw text from sections of resume

#     :param text: Raw text of resume
#     :return: dictionary of entities
#     '''
#     text_split = [i.strip() for i in text.split('\n')]
#     # sections_in_resume = [i for i in text_split if i.lower() in sections]
#     entities = {}
#     key = False
#     for phrase in text_split:
#         if len(phrase) == 1:
#             p_key = phrase
#         else:
#             p_key = set(phrase.lower().split()) & set(cs.RESUME_SECTIONS)
#         try:
#             p_key = list(p_key)[0]
#         except IndexError:
#             pass
#         if p_key in cs.RESUME_SECTIONS:
#             entities[p_key] = []
#             key = p_key
#         elif key and phrase.strip():
#             entities[key].append(phrase)
    
#     # entity_key = False
#     # for entity in entities.keys():
#     #     sub_entities = {}
#     #     for entry in entities[entity]:
#     #         if u'\u2022' not in entry:
#     #             sub_entities[entry] = []
#     #             entity_key = entry
#     #         elif entity_key:
#     #             sub_entities[entity_key].append(entry)
#     #     entities[entity] = sub_entities

#     # pprint.pprint(entities)

#     # make entities that are not found None
#     # for entity in cs.RESUME_SECTIONS:
#     #     if entity not in entities.keys():
#     #         entities[entity] = None 
#     return entities
# # print(extract_entity_sections(retext))

# def extract_email(text):
#     '''
#     Helper function to extract email id from text

#     :param text: plain text extracted from resume file
#     '''
#     email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
#     if email:
#         try:
#             return email[0].split()[0].strip(';')
#         except IndexError:
#             return None
# # print(extract_email(retext))



# def extract_name(text, matcher):
#     '''
#     Helper function to extract name from spacy nlp text

#     :param nlp_text: object of `spacy.tokens.doc.Doc`
#     :param matcher: object of `spacy.matcher.Matcher`
#     :return: string of full name
#     '''
#     pattern = [cs.NAME_PATTERN]
    
#     matcher.add('NAME', None, *pattern)
    
#     matches = matcher(text)
    
#     for match_id, start, end in matches:
#         span = text[start:end]
#         return span.text
# # print(extract_name(retext, matcher))

# def extract_mobile_number(text):
#     '''
#     Helper function to extract mobile number from text

#     :param text: plain text extracted from resume file
#     :return: string of extracted mobile numbers
#     '''
#     # Found this complicated regex on : https://zapier.com/blog/extract-links-email-phone-regex/
#     phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
#     if phone:
#         number = ''.join(phone[0])
#         if len(number) > 10:
#             return '+' + number
#         else:
#             return number
# # print(extract_mobile_number(retext))

# def extract_skills(nlp_text, noun_chunks):
#     '''
#     Helper function to extract skills from spacy nlp text

#     :param nlp_text: object of `spacy.tokens.doc.Doc`
#     :param noun_chunks: noun chunks extracted from nlp text
#     :return: list of skills extracted
#     '''
#     tokens = [token.text for token in nlp_text if not token.is_stop]
#     data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'skills.csv')) 
#     skills = list(data.columns.values)
#     skillset = []
#     # check for one-grams
#     for token in tokens:
#         if token.lower() in skills:
#             skillset.append(token)
    
#     # check for bi-grams and tri-grams
#     for token in noun_chunks:
#         token = token.text.lower().strip()
#         if token in skills:
#             skillset.append(token)
#     return [i.capitalize() for i in set([i.lower() for i in skillset])]
# # print(extract_skills(text))

# def cleanup(token, lower = True):
#     if lower:
#        token = token.lower()
#     return token.strip()

# def extract_education(nlp_text):
#     '''
#     Helper function to extract education from spacy nlp text

#     :param nlp_text: object of `spacy.tokens.doc.Doc`
#     :return: tuple of education degree and year if year if found else only returns education degree
#     '''
#     edu = {}
#     # Extract education degree
#     for index, text in enumerate(nlp_text):
#         for tex in text.split():
#             tex = re.sub(r'[?|$|.|!|,]', r'', tex)
#             if tex.upper() in cs.EDUCATION and tex not in cs.STOPWORDS:
#                 edu[tex] = text + nlp_text[index + 1]

#     # Extract year
#     education = []
#     for key in edu.keys():
#         year = re.search(re.compile(cs.YEAR), edu[key])
#         if year:
#             education.append((key, ''.join(year.group(0))))
#         else:
#             education.append(key)
#     return education
# # print(extract_education(retext))

# def extract_experience(resume_text):
#     '''
#     Helper function to extract experience from resume text

#     :param resume_text: Plain resume text
#     :return: list of experience
#     '''
#     wordnet_lemmatizer = WordNetLemmatizer()
#     stop_words = set(STOPWORDS.words('english'))

#     # word tokenization 
#     word_tokens = nltk.word_tokenize(resume_text)

#     # remove stop words and lemmatize  
#     filtered_sentence = [w for w in word_tokens if not w in stop_words and wordnet_lemmatizer.lemmatize(w) not in stop_words] 
#     sent = nltk.pos_tag(filtered_sentence)

#     # parse regex
#     cp = nltk.RegexpParser('P: {<NNP>+}')
#     cs = cp.parse(sent)
    
#     # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
#     #     print(i)
    
#     test = []
    
#     for vp in list(cs.subtrees(filter=lambda x: x.label()=='P')):
#         test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))

#     # Search the word 'experience' in the chunk and then print out the text after it
#     x = [x[x.lower().index('experience') + 10:] for i, x in enumerate(test) if x and 'experience' in x.lower()]
#     return x

# def extract_competencies(text, experience_list):
#     '''
#     Helper function to extract competencies from resume text

#     :param resume_text: Plain resume text
#     :return: dictionary of competencies
#     '''
#     experience_text = ' '.join(experience_list)
#     competency_dict = {}

#     for competency in cs.COMPETENCIES.keys():
#         for item in cs.COMPETENCIES[competency]:
#             if (item, experience_text):
#                 if competency not in competency_dict.keys():
#                     competency_dict[competency] = [item]
#                 else:
#                     competency_dict[competency].append(item)
    
#     return competency_dict

# def extract_measurable_results(text, experience_list):
#     '''
#     Helper function to extract measurable results from resume text

#     :param resume_text: Plain resume text
#     :return: dictionary of measurable results
#     '''

#     # we scan for measurable results only in first half of each sentence
#     experience_text = ' '.join([text[:len(text) // 2 - 1] for text in experience_list])
#     mr_dict = {}

#     for mr in cs.MEASURABLE_RESULTS.keys():
#         for item in cs.MEASURABLE_RESULTS[mr]:
#             if string_found(item, experience_text):
#                 if mr not in mr_dict.keys():
#                     mr_dict[mr] = [item]
#                 else:
#                     mr_dict[mr].append(item)
    
#     return mr_dict

