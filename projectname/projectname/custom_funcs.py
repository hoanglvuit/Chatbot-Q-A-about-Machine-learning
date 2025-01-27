import re
def remove_incorrect_pattern(document,incorrect_title_pattern) : 
    document = document.casefold() 
    document = re.sub(incorrect_title_pattern, '',document) 
    return document

def split_chapter(document, sign) : 
    parts = re.split(sign, document, flags = re.MULTILINE) 
    return parts 

def get_title(document) : 
    return document.splitlines()[0]

def get_keyword(document) : 
    return [re.sub(r"\d+", "", line[1:].strip()).strip() for line in re.findall(r"^#.*", document, re.MULTILINE)] 




  

