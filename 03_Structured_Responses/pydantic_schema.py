from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Schema(BaseModel):
    key_themes: list[str]=Field(description="write key themes discussed in the review in a list")
    summary: str=Field(description="brief summary of the review")
    sentiment: Literal["Positive", "Negative"] = Field(description="return sentiment of the review")
    pros: Optional[list[str]] = Field(default=None, description="write all pros in a list")
    cons: Optional[list[str]] = Field(default=None, description="write all cons in a list")
    name: Optional[str] = Field(default=None, description="write name of the person who wrote the review, generally found at the end of the review")

structured_model = model.with_structured_output(Schema, strict=True)

prompt = """Google's Pixel smartphones have traditionally focused on software, AI features, and camera quality rather than raw performance. While the Tensor chipsets powering Pixel devices have improved over the years, they have generally lagged behind competing flagship processors from Apple, Qualcomm, and Samsung in benchmark tests.

Recent benchmark leaks suggest that the upcoming Google Pixel 10 series may continue this trend and could even perform worse than the Pixel 9 series in certain graphics-related workloads. According to reports circulating online, a Geekbench benchmark listing for the Google Pixel 10 Pro revealed a surprisingly low GPU score of approximately 3,707 points.

For context, benchmark scores are commonly used to compare device performance across different smartphones. In GPU benchmarks, higher scores generally indicate better graphics performance and smoother handling of demanding applications such as gaming, video rendering, and graphical AI workloads.

The leaked result becomes more concerning when compared with competing devices. The Google Pixel 9 Pro reportedly achieved a GPU score of around 9,023 points. Meanwhile, flagship competitors such as the Samsung Galaxy S25 Plus and Apple's iPhone 16 Pro scored approximately 26,333 and 33,374 points respectively. This places the Pixel 10 Pro significantly behind its direct competitors in terms of graphics processing capability.

However, GPU performance represents only one aspect of a smartphone's overall performance profile. Other benchmark results suggest that the Pixel 10 Pro and Pixel 10 Pro XL may still offer improvements in CPU performance. Reports indicate single-core benchmark scores around 2,329 and multi-core scores around 6,358 for the Pixel 10 Pro XL. These numbers are noticeably higher than the Pixel 9 Pro XL, which achieved roughly 1,948 in single-core tests and 4,530 in multi-core tests.

This suggests that while graphical performance may be weaker than expected, general computing performance could still see meaningful improvements. Everyday tasks such as web browsing, social media usage, multitasking, productivity applications, and AI-assisted features may perform smoothly despite weaker GPU benchmarks.

Gaming enthusiasts, however, may have reason to be concerned. Modern mobile games increasingly rely on powerful GPUs to deliver high frame rates, advanced visual effects, and stable performance. A significantly lower GPU score could lead to reduced graphics settings, lower frame rates, or less consistent gaming experiences when compared with competing flagship devices.

Google has emphasized that the Pixel 10 series is designed around practical user experiences rather than benchmark leadership. The company claims that improvements in software optimization, machine learning acceleration, and AI-powered features will make the devices feel faster in real-world usage. Features such as on-device AI processing, smarter voice assistants, advanced photography tools, and productivity enhancements may therefore become the primary selling points of the Pixel 10 lineup.

Industry analysts remain divided on what these benchmark leaks mean. Some experts argue that synthetic benchmarks do not always reflect real-world performance and that software optimization can compensate for hardware limitations. Others believe that such a large performance gap is difficult to ignore, especially in a highly competitive premium smartphone market.

Consumers interested primarily in photography, AI capabilities, and Google's software ecosystem may still find the Pixel 10 series appealing. On the other hand, users who prioritize gaming, graphics-intensive applications, or maximum performance-per-dollar may prefer alternatives from Apple, Samsung, or other Android manufacturers.

Overall, the leaked benchmark results paint a mixed picture of the Pixel 10 series. The devices appear to offer improved CPU performance and enhanced AI capabilities but may struggle to compete with rival flagship smartphones when it comes to graphics performance. Until official benchmarks and independent reviews become available, these reports should be treated as preliminary information rather than definitive conclusions."""

response = structured_model.invoke(prompt)
print(response)
