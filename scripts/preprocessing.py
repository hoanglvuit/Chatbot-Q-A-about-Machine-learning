from projectname.config import * 
from projectname.custom_funcs import * 

with open(md_data) as file : 
    document = file.read() 

# remove incorrect title or partern 
document = remove_incorrect_pattern(document, r'---\n# chương (\d+). (.*?)\n\n') 
document = remove_incorrect_pattern(document, r'\n\nmachine learning cơ bản\n\nhttps://machinelearningcoban.com\n') 
document = remove_incorrect_pattern(document, r'\n\nmachine learning cơ bản https://machinelearningcoban.com\n') 
document = remove_incorrect_pattern(document, r'machine learning cơ bản\n\nhttps://machinelearningcoban.com\n') 
document = remove_incorrect_pattern(document,  r'machine learning cơ bản\n\nhttps://machinelearningcoban.com')  

# split into chapters 
Documets = split_chapter(document, "chương\s+\d+\n\n") 

# get title of each chapter
titles = [] 
for doc in Documets : 
    titles.append(get_title(doc)) 

# get keywords for each chapter 
keywords = [] 
for doc in Documets : 
    keywords.append(get_keyword(doc)) 
keywords[1] = keywords[1][:16]


