from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

json_schema = {"title":"Review",
               "type":"object",
               "properties":{"key_themes":{"type":"array",
                                           "items":{"type":"string"},
                                           "description":"write all key themes discussed in the review in a list"},
                            "summary":{"type":"string",
                                       "description":"brief summary of the review"},
                            "sentiment":{"type":"string",
                                         "enum":["Positive", "Negative"],
                                         "description":"return sentiment of the review in either Positive, Negative or Neutral"},
                            "pros":{"type":["array", "null"],
                                    "items":{"type":"string"},
                                    "description":"write all pros in the list"},
                            "cons":{"type":["array", "null"],
                                    "items":{"type":"string"},
                                    "description":"write all cons in the list"},
                            "name":{"type":["string", "null"],
                                    "description":"write name of the reviewer"}
                            },
                            "required":["key_themes", "summary", "sentiment"]
              }

structured_model = model.with_structured_output(json_schema, strict=True)

prompt = """Google's Pixel smartphones have traditionally focused on software, AI features, and camera quality rather than raw performance. While the Tensor chipsets powering Pixel devices have improved over the years, they have generally lagged behind competing flagship processors from Apple, Qualcomm, and Samsung in benchmark tests.

The leaked result becomes more concerning when compared with competing devices. The Google Pixel 9 Pro reportedly achieved a GPU score of around 9,023 points. Meanwhile, flagship competitors such as the Samsung Galaxy S25 Plus and Apple's iPhone 16 Pro scored approximately 26,333 and 33,374 points respectively. This places the Pixel 10 Pro significantly behind its direct competitors in terms of graphics processing capability.

However, GPU performance represents only one aspect of a smartphone's overall performance profile. Other benchmark results suggest that the Pixel 10 Pro and Pixel 10 Pro XL may still offer improvements in CPU performance. Reports indicate single-core benchmark scores around 2,329 and multi-core scores around 6,358 for the Pixel 10 Pro XL. These numbers are noticeably higher than the Pixel 9 Pro XL, which achieved roughly 1,948 in single-core tests and 4,530 in multi-core tests.

Gaming enthusiasts, however, may have reason to be concerned. Modern mobile games increasingly rely on powerful GPUs to deliver high frame rates, advanced visual effects, and stable performance. A significantly lower GPU score could lead to reduced graphics settings, lower frame rates, or less consistent gaming experiences when compared with competing flagship devices.

Google has emphasized that the Pixel 10 series is designed around practical user experiences rather than benchmark leadership. The company claims that improvements in software optimization, machine learning acceleration, and AI-powered features will make the devices feel faster in real-world usage. Features such as on-device AI processing, smarter voice assistants, advanced photography tools, and productivity enhancements may therefore become the primary selling points of the Pixel 10 lineup.

Industry analysts remain divided on what these benchmark leaks mean. Some experts argue that synthetic benchmarks do not always reflect real-world performance and that software optimization can compensate for hardware limitations. Others believe that such a large performance gap is difficult to ignore, especially in a highly competitive premium smartphone market.

Consumers interested primarily in photography, AI capabilities, and Google's software ecosystem may still find the Pixel 10 series appealing. On the other hand, users who prioritize gaming, graphics-intensive applications, or maximum performance-per-dollar may prefer alternatives from Apple, Samsung, or other Android manufacturers."""

response = structured_model.invoke(prompt)
print(response)
