from lambeq import SpacyTokeniser, BobcatParser, spiders_reader, cups_reader, stairs_reader, TreeReader, TreeReaderMode
sentance = 'hello my name is be, how are you?'
# Break sentance into tokens
def tokens():
    tokeniser = SpacyTokeniser()
    tokens = tokeniser.tokenise_sentence(sentance)
    print(tokens)

#create tokens diagram
def tokens_diagram():
    parser = BobcatParser(verbose='suppress')
    diagram = parser.sentence2diagram(tokens, tokenised=True)
    diagram.draw(figsize=(23,4), fontsize=12)

# Spyder Diagram 
def spyder_diagram():
    spiders_diagram = spiders_reader.sentence2diagram(sentance)
    spiders_diagram.draw(figsize=(10,6), fontsize=12)

def cups_diagram():
    cups_diagram = cups_reader.sentence2diagram(sentance)
    cups_diagram.draw(figsize=(12,2), fontsize=10)

def stairs_diagram():
    stairs_diagram = stairs_reader.sentence2diagram(sentance)
    stairs_diagram.draw(figsize=(12,5), fontsize=12)

def Tree_diagram():
    reader = TreeReader()
    tree_diagram = reader.sentence2diagram(sentance)
    tree_diagram.draw(figsize=(12,5), fontsize=12)

def Tree_diagram_mode():
    reader = TreeReader(mode=TreeReaderMode.RULE_ONLY)
    tree_diagram = reader.sentence2diagram(sentance)
    tree_diagram.draw(figsize=(12,5), fontsize=12)
