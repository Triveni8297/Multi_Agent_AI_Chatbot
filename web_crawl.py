# To install: pip install tavily-python
from tavily import TavilyClient
client = TavilyClient("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")
response = client.crawl(
    url="https://www.innomatics.in/",
    max_depth=4,
    limit=50,
    instructions="extract all content from the given URL",
    extract_depth="advanced"
)
print(response)

# print raw_content from res
whole_text = ""
results = response["results"]
print(f"Total results: {len(results)}")
for result in results:
    print(result["raw_content"])
    print("***********************\n\n\n\n")
    whole_text += result["raw_content"] + "\n\n\n"

# save response to a file
with open("crawled_data.txt", "w", encoding="utf-8") as file:
    file.write(whole_text)