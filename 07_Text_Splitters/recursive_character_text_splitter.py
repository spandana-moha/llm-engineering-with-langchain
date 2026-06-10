from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Sachin Tendulkar is known as the "God of Cricket" and scored 100 international centuries.

Virat Kohli is regarded as one of the greatest batters of the modern era and has scored more than 80 international centuries.

MS Dhoni is one of the most successful captains in cricket history. Under his leadership, India won the 2007 T20 World Cup, the 2011 ODI World Cup, and the 2013 Champions Trophy.

Rohit Sharma holds the record for the highest individual score in ODI cricket with 264 runs.

Muttiah Muralitharan from Sri Lanka holds the record for the most wickets in Test cricket with 800 wickets."""

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)

result = splitter.split_text(text)
print(result)
