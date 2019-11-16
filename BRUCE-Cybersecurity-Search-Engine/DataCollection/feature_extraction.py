# -*- coding: utf-8 -*-
from nltk.tokenize import sent_tokenize
from BERT_NER.bert import Ner
from pathlib import Path
from tqdm import trange
import pandas as pd
import tldextract
import datetime
import csv
import re

# Set this option to None unless you know what you are doing
pd.options.mode.chained_assignment = None

# Find phone number with regex (US number only)
def extract_phone_numbers(string):
    r = re.compile(r'(?:(?:(?:\+)?1[\-\s\.])?(?:\s?\()?(?:[2-9][0-8][0-9])(?:\))?(?:[\s|\-|\.])?)(?:(?:(?:[2-9][0-9|A-Z][0-9|A-Z])(?:[\s|\-|\.])?)(?:[0-9|A-Z][0-9|A-Z][0-9|A-Z][0-9|A-Z]))')
    phone_numbers = r.findall(string)
    return phone_numbers

# Find domain with tldextract
def domain_finder(url):
    return tldextract.extract(url).domain

# Find year with regex
def year_finder(text):
    r = re.compile(r'20[012]\d')
    years = r.findall(text)
    return years

# Find Semester with regex
def semester_finder(text):
    r = re.compile(r"[Ff]all|[Ss]pring|[Ww]inter|[Ss]ummer|[Aa]utumn")
    semesters = r.findall(text)
    return semesters

# Find final exam with regex
def final_finder(text):
    r = re.compile(r"[fF]inal\s*(?:[Ee]xam(?:s)?)?")
    finals = r.findall(text)
    if any(finals):
        return True
    else:
        return False

# Find midterm exam with regex
def midterm_finder(text):
    r = re.compile(r"[Mm]idterm\s*(?:[Ee]xam(?:s)?)?")
    midterms = r.findall(text)
    if any(midterms):
        return True
    else:
        return False

# Find project with regex
def project_finder(text):
    r = re.compile(r"[Pp]rojects?")
    projects = r.findall(text)
    if any(projects):
        return True
    else:
        return False

# Find homework, assignment, and presentation with regex
def homework_finder(text):
    r = re.compile(r"[Hh]omeworks?|[Aa]ssignments?|[Pp]resentations?")
    homeworks = r.findall(text)
    if any(homeworks):
        return True
    else:
        return False

# Find quiz with regex
def quiz_finder(text):
    r = re.compile(r"[Qq]uiz(?:zes)?")
    quizzes = r.findall(text)
    if any(quizzes):
        return True
    else:
        return False

# The entity finding pattern is compatible with BOI1 and BOI2
def entity_finder(result_list):
    tags = [
        "Course_title",
        "Course_number",
        "Email",
        "Textbook",
        "Person"
    ]
    entities = {key: [] for key in tags}
    find_subsequent = False
    current_tag = None
    i = 0
    for i in range(len(result_list)):
        if result_list[i]['tag'] != "O" and result_list[i]['tag'] != '[SEP]':
            if result_list[i]['tag'].startswith("B"):
                entities[result_list[i]['tag'].split("-")[1]].append(result_list[i]['word'])
                current_tag = result_list[i]['tag'].split("-")[1]
                find_subsequent = True
            else:
                if find_subsequent == True and current_tag == result_list[i]['tag'].split("-")[1]:
                    if current_tag in ["Course_number", "Email"]:
                        entities[current_tag][-1] = entities[current_tag][-1] + result_list[i]['word']
                    else:
                        entities[current_tag][-1] = entities[current_tag][-1] + " " + result_list[i]['word']
                else:
                    current_tag = result_list[i]['tag'].split("-")[1]
                    find_subsequent = True
                    entities[result_list[i]['tag'].split("-")[1]].append(result_list[i]['word'])
        else:
            find_subsequent = False
            current_tag = None
    return entities

# Find entities with NER
def get_entity(text):
    names = []
    textbooks = []
    emails = []
    course_titles = []
    course_numbers = []
    lines = text.split("\n")
    for line in lines:
        sentences = sent_tokenize(line.strip())
        for sentence in sentences:
            result_list = model.predict(sentence.strip())
            entities = entity_finder(result_list)
            if entities['Person']:
                names.extend(entities['Person'])
            if entities['Email']:
                emails.extend(entities['Email'])
            if entities['Course_title']:
                course_titles.extend(entities['Course_title'])
            if entities['Course_number']:
                course_numbers.extend(entities['Course_number'])
            if entities['Textbook']:
                textbooks.extend(entities['Textbook'])
    return names, textbooks, emails, course_titles, course_numbers

# To see if the professor has these elements listed for the course
def get_tests(text):
    fe = final_finder(text)
    mid = midterm_finder(text)
    proj = project_finder(text)
    quiz = quiz_finder(text)
    pres = [False, False, False, False]
    if fe:
        pres[0] = True
    if mid:
        pres[1] = True
    if proj:
        pres[2] = True
    if quiz:
        pres[3] = True
    return pres


if __name__ == "__main__":
    model = Ner("BERT_NER/out_!x_course_crf/")
    url_folder = Path("../Course_Collected/")
    corpus_folder = Path("../Data/Corpus2/")

    # Read url list
    df = pd.read_csv(url_folder / "Final.csv", error_bad_lines=False, encoding='utf-8').copy()
    for i in trange(len(df)):
        try:
            url = str(df.URL[i])
            with open(corpus_folder / (str(i) + ".txt"), "r", encoding="utf-8") as file:
                text = file.read()

            df.DOMAIN[i] = (domain_finder(url))

            names, textbooks, emails, course_titles, course_numbers = get_entity(text)

            if course_numbers and course_titles:
                df.TITLE[i] = (course_numbers[0] + " " + course_titles[0])
            elif course_numbers:
                df.TITLE[i] = (course_numbers[0])
            elif course_titles:
                df.TITLE[i] = (course_titles[0])
            else:
                df.TITLE[i] = ("None")

            year = year_finder(text)
            if year:
                year = year[0]
                df.YEAR[i]=(year)
            else:
                df.YEAR[i] = ("None")

            if names:
                name = names[0]
                df.PROF[i] = (name)
            else:
                df.PROF[i] = ("None")

            phone = extract_phone_numbers(text)
            if phone:
                phone = phone[0]
                df.PHONE[i]=(phone)
            else:
                df.PHONE[i] = ("None")

            if emails:
                email = emails[0]
                df.EMAIL[i] = (email)
            else:
                df.EMAIL[i] = ("None")

            df.CONTENT[i]=([text])

            if textbooks:
                textbook = " & ".join([t for t in textbooks])
                df.TEXTBOOK[i] = (textbook)
            else:
                df.TEXTBOOK[i] = ("None")

            df.TEST[i]=(get_tests(text))
        except Exception as e:
            print(i)
            print(e)
            continue
    now = datetime.datetime.now()
    df.to_csv(str(now)[:10] + '.csv', index = False)