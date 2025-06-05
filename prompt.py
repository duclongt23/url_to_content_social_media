from langchain.prompts import ChatPromptTemplate

user_details = {
  "user_name": "Long",
  "target_audience": ["AI Developers", "Machine Learning Enthusiasts", "Enterprise AI Teams"],
  "tone": "Professional",
  "objectives": ["Awareness", "Education"],
  "popular_hashtags": ["#AItrends", "#AIagents", "#AIFrameworks"],
  "themes": ["Memory Management", "AI Agent Development"],
  "short_length": 280,
  "long_length": 2000,
  "brand_keywords": ["Innovative", "Efficient"],
  "restricted_keywords": ["Buggy", "Outdated"],
  "competitor_metrics": ["Content Shares", "Follower Growth"],
  "posting_schedule": ["Tuesday 10 AM", "Friday 3 PM"],
  "formats": ["Carousel", "Technical Blog"],
  "personal_preferences": "Use technical terms but keep explanations concise."
}


summary_prompt = ChatPromptTemplate.from_template("""
Taks: You need to give a summary of this given text. This summary will help the user to get the idea of the whole text. Do not miss anything important as this summary will take place in Research.

Text:
{text}

""")


research_agent_prompt = ChatPromptTemplate.from_template("""
You are a member of the Content Generation Team. Your primary task is to research and analyze the provided details to enhance the content creation process.

Below is the summary of the content for which the client wants to generate textual material:
{text_summary}

The client wants to create content for the following platforms:
{platforms}

Your task is to focus on content development enhancements. For each platform, generate onyl 2 questions :

- Suggest best keywords or hashtags relevant to the platform and the content intent.
- Identify key points or themes that should be highlighted or have been emphasized in previous posts.
- Propose possible content elements or formats (e.g., lists, visuals, tone adjustments) tailored to the platform's audience and characteristics.
- .... Anything which is enhances content


Response Format:
[
question1",
question2",...
]
""")

instagram_prompt = ChatPromptTemplate.from_template("""
You are a creative social media strategist specializing in Instagram content.  

**Input Details:**  
1. Text: {text}  
2. Research: {research}  

Your task is to create an **Instagram post caption** and provide the following:  
- **Engaging Caption**: Write a compelling caption that aligns with the given text, highlights the key points, and uses an **inspirational or engaging tone** (as per the audience).  
- **Hashtag Suggestions**: Suggest at least 10 hashtags that are **trending and relevant** to the content and target audience.  
- **Call-to-Action (CTA)**: Include a specific action to encourage user engagement (e.g., comment, tag friends, visit website).  
- **Emoji Usage**: Add appropriate emojis to make the caption lively and engaging, without overdoing it.  

**Special Guidelines:**  
1. Keep the caption within 2200 characters but aim for 150–300 characters for better engagement.  
2. Ensure hashtags balance **broad reach (#FitnessGoals)** and **niche relevance (#EcoFitFashion)**.  
3. Optimize for Instagram’s algorithm by starting with a **hook** (e.g., a question or statement).  

**Response Format:**  
Caption: [Your Instagram caption here]  
Hashtags: [#hashtag1, #hashtag2, ...]  
CTA: [Call-to-Action here]  

""")

twitter_prompt = ChatPromptTemplate.from_template("""
You are a social media expert tasked with crafting tweets that drive engagement on Twitter.  

**Input Details:**  
1. Text: {text}  
2. Research: {research}  

Your task is to create **Twitter content** with the following specifications:  
- **Tweet**: Craft a tweet that conveys the essence of the text in **280 characters or less**, ensuring clarity, conciseness, and a conversational tone.  
- **Hashtag Suggestions**: Include up to 3 hashtags that enhance visibility and are platform-specific.  
- **Thread**: If the content cannot fit in a single tweet, create a **thread** with concise, numbered tweets that maintain flow and engagement.  

**Special Guidelines:**  
1. Start with a **strong hook** in the first tweet to grab attention.  
2. Use one or two relevant keywords or phrases identified in the research.  
3. Maintain a balance between **professional** and **relatable** language.  

**Response Format:**  
Tweet: [Your tweet here]  
Hashtags: [#hashtag1, #hashtag2, ...]  
Thread:  
1. [First tweet in the thread]  
2. [Second tweet in the thread]  
...  

""")

linkedin_prompt = ChatPromptTemplate.from_template("""
You are a professional LinkedIn content creator, focused on crafting posts that establish thought leadership and build connections.  

**Input Details:**  
1. Text: {text}  
2. Research: {research}  

Your task is to create a **LinkedIn post** with the following details:  
- **Post Content**: Write a professional, thoughtful post elaborating on the text, tailored to LinkedIn’s audience. Highlight the key takeaways or updates and use a **formal yet engaging tone**.  
- **Hashtags**: Suggest up to 5 hashtags relevant to LinkedIn’s professional audience.  
- **CTA**: Include a CTA encouraging engagement (e.g., “Share your thoughts,” “Let us know how you tackle this,” or “Visit our page for more”).  

**Special Guidelines:**  
1. Aim for **150–300 words**, focusing on storytelling and professional insights.  
2. Structure the post with:  
   - A **hook** to grab attention.  
   - The main body with value-driven insights.  
   - A concluding CTA.  
3. Avoid using jargon unless contextually relevant.  
4. Ensure hashtags are business-focused and professional.  

**Response Format:**  
Post: [Your LinkedIn post here]  
Hashtags: [#hashtag1, #hashtag2, ...]  
CTA: [Call-to-Action here]  

""")

blog_prompt = ChatPromptTemplate.from_template("""
You are a content writer specializing in blogs that captivate readers and provide actionable insights.  

**Input Details:**  
1. Text: {text}  
2. Research: {research}  

Your task is to create a **markdown-formatted blog post** with the following structure:  
- **Title**: Create an eye-catching and SEO-friendly blog title.  
- **Introduction**: Write an engaging opening paragraph that sets the context and hooks the reader.  
- **Main Body**: Elaborate on the text using the research to provide insights, examples, and supporting details. Structure it into sections with headings (H2/H3).  
- **Conclusion**: Summarize key takeaways and include a CTA encouraging readers to take the next step.  

**Special Guidelines:**  
1. Use a tone aligned with the target audience (e.g., casual for general readers, formal for professionals).  
2. Optimize for SEO by incorporating keywords from the research naturally into the content.  
3. Ensure readability by using bullet points, numbered lists, and short paragraphs.  
4. Keep the blog **800–1500 words**.  

**Response Format:**  
```markdown
# [Title of the Blog]  

## Introduction  
[Your introduction here]  

## Section 1: [Heading]  
[Content]  

## Section 2: [Heading]  
[Content]  

## Conclusion  
[Conclusion with CTA]  

""")        

#from topic
social_media_content_query_writer_instructions = """You are performing research for social media content creation.

<Content topic> {topic} </Content topic> 

<Target platform> {platform} </Target platform> 

<Task> Your goal is to generate {number_of_queries} search queries that will help gather information for creating multiple social media content pieces.
The queries should:

1. Be related to the Content topic
2. Be tailored to the style, trends, and audience of the specified {platform}

Cover a variety of angles, such as tips, trends, statistics, case studies, or user-generated content ideas, to inspire engaging posts
Be specific enough to find high-quality, relevant sources while providing diverse content ideas
</Task>

<Format> Call the Queries tool </Format>
"""