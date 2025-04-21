## Question-Answer Pair Generation Workflow

The goal is to generate a dataset of question-answer pairs for prospective students at SUTD.

### One-shot Theme Generation Prompt Template

```
Generate a JSON object containing 5-10 key themes or points of interest that prospective university students commonly inquire about when considering a university experience. The keys of the JSON object should be sequential integers starting from 0, and the values should be descriptive strings representing the themes.

Example of the desired output format:
{
  "0": "Student Life",
  "1": "Extracurriculars",
  "2": "Academics",
  "3": "Admissions"
}

The output should be in a JSON-like format within the curly braces.
```


### Few-shot Categorical Question Generation Prompt Template
```
You are helping to generate lexicon-rich question templates for prospective students exploring the Singapore University of Technology and Design (SUTD). Each question should contain a lexicon placeholder (e.g., {CLUB}, {PROGRAM_TYPE}, {SEMESTER_YEAR}) that can later be substituted with concrete values retrieved from the web.

Here are some themes relevant to prospective students, along with sample question templates:

Theme: Student Life and Culture  
{
  "Social Events": "What kind of social events or traditions does SUTD host during {SEMESTER_OR_PERIOD}?",
  "Campus Culture": "What is student culture like for those studying in {PROGRAM_TYPE} at SUTD?",
  "Arts and Culture": "What opportunities are there for students interested in {ARTS_ACTIVITY} at SUTD?"
}

Theme: Admissions and Applications  
{
  "Application Requirements": "What are the application requirements for {PROGRAM_TYPE} programs at SUTD?",
  "Essay Guidance": "Do you have any advice for writing the application essay for {PROGRAM_TYPE} at SUTD?",
  "Standardized Tests": "What are the policies regarding {TEST_TYPE} scores when applying to SUTD?",
  "Application Deadline": "What is the application deadline for entry in {SEMESTER_YEAR}?"
}

Theme: {YOUR_THEME}  
{
  "{SUB_THEME_1_KEY}": "{SUB_THEME_1_QUERY}",
  "{SUB_THEME_2_KEY}": "{SUB_THEME_2_QUERY}",
  "{SUB_THEME_3_KEY}": "{SUB_THEME_3_QUERY}",
  "{SUB_THEME_4_KEY}": "{SUB_THEME_4_QUERY}"
}

Generate 3–5 relevant sub-themes (keys) and corresponding question templates (values) for the theme: {YOUR_THEME}. Use placeholders like {COURSE}, {FACILITY}, {HOSTEL}, {CLUB}, {SEMESTER_YEAR}, etc., to keep the questions lexicon-ready. The output should be in a JSON-like format within the curly braces.
```

### Few-shot Lexicon-based Web Search Prompt Template

```
You are an expert in generating effective web search queries for finding information related to the Singapore University of Technology and Design (SUTD). Your goal is to create a concise web search query that will retrieve possible values for the lexicon placeholder `{LEXICON_PLACEHOLDER}` within the context of SUTD.

Here are a few examples:

**Example 1:**
Question Template: "What does {CLUB} do in SUTD?"
LEXICON_PLACEHOLDER: {CLUB}
Search Query: "SUTD student clubs organizations list"

**Example 2:**
Question Template: "Who are the professors teaching {COURSE} at SUTD?"
LEXICON_PLACEHOLDER: {COURSE}
Search Query: "SUTD course catalog undergraduate graduate"

Now, generate a search query for the following:

Question Template: "{QUESTION_TEMPLATE_TO_SEARCH}"
LEXICON_PLACEHOLDER: {LEXICON_PLACEHOLDER_TO_FILL}
Search Query:
```
**Replace the placeholders:**
* `{QUESTION_TEMPLATE_TO_SEARCH}`: Insert the question template you are working with.
* `{LEXICON_PLACEHOLDER_TO_FILL}`: Specify the placeholder you want to fill.

### Few-shot Lexicon Extraction Prompt Template

```
You are an expert in information extraction. Your goal is to parse the provided text and identify specific values that can fill the placeholder `{LEXICON_PLACEHOLDER}` in the following question template:

"{QUESTION_TEMPLATE}"

The following is the text retrieved from a web search (or other source) that may contain the information you need:

--- START OF CONTEXT ---
{RETRIEVED_CONTEXT}
--- END OF CONTEXT ---

Based on this context, please extract all relevant and distinct values that can be used to replace the placeholder `{LEXICON_PLACEHOLDER}` in the question template. List each potential value on a new line. If no relevant values are found in the context, please state "No relevant values found."

Example 1:
Question Template: "What does {CLUB} do in SUTD?"
LEXICON_PLACEHOLDER: {CLUB}
Retrieved Context: "SUTD offers a wide range of student clubs including the SUTD Robotics Club, SUTD Design Society, SUTD Photography Club, and the SUTD Debate Club..."
Extracted Values:
SUTD Robotics Club
SUTD Design Society
SUTD Photography Club
SUTD Debate Club

Example 2:
Question Template: "Who are the professors teaching {COURSE} at SUTD?"
LEXICON_PLACEHOLDER: {COURSE}
Retrieved Context: "The undergraduate course catalog lists Introduction to Programming taught by Prof. Lee, and Calculus I taught by Dr. Tan..."
Extracted Values:
Introduction to Programming
Calculus I

Now, apply this to the following:

Question Template: "{QUESTION_TEMPLATE_TO_PARSE}"
LEXICON_PLACEHOLDER: {LEXICON_PLACEHOLDER_TO_EXTRACT}
Retrieved Context:
--- START OF CONTEXT ---
{YOUR_RETRIEVED_CONTEXT}
--- END OF CONTEXT ---

Extracted Values:
```

**Replace the placeholders:**
* `{QUESTION_TEMPLATE_TO_PARSE}`: Insert the question template you are working with.
* `{LEXICON_PLACEHOLDER_TO_EXTRACT}`: Specify the placeholder you want to fill.
* `{YOUR_RETRIEVED_CONTEXT}`: Paste the text you obtained from your web search (or other information source).


### Zero-shot Grounded Question-Answer Pair Generation Prompt Template

```
You are a university assistant for Singapore University of Technology and Design (SUTD). Using the question below and a retrieved context, generate a concise and informative answer appropriate for a prospective student to SUTD.

Question: "{QUESTION_STRING}"

Context:
--- START OF CONTEXT ---
{RETRIEVED_CONTEXT}
--- END OF CONTEXT ---

Answer:
```

**Replace the placeholders:**
* `{QUESTION_STRING}`: Insert the question you are working with.

### Workflow Diagram
The intended framework to use is Langchain, with our base LLM being Google Gemini 2.0 Flash.
```mermaid
graph TD
  A[Theme Generation] --> B[Question Template Generation]
  B --> C[Search Query Generation]
  C --> D[Web Search (Tool)]
  D --> E[Lexicon Extraction]
  E --> F[Grounded QA Generation]
  F --> G[Save JSONL QA Dataset]
```

Below is a high-level implementation plan and code snippets for each stage of your QA pair generation workflow, tailored for use with Langchain and Google Gemini 2.0 Flash. The code is written in Python and assumes you have access to the necessary LLM APIs and web search tools.

---

## **1. Theme Generation**

**Prompt Template:** (as provided)

**Langchain Chain:**

```python
from langchain.prompts import PromptTemplate
from langchain.llms import Gemini

theme_prompt = PromptTemplate(
    input_variables=[],
    template="""
Generate a JSON object containing 5-10 key themes or points of interest that prospective university students commonly inquire about when considering a university experience. The keys of the JSON object should be sequential integers starting from 0, and the values should be descriptive strings representing the themes.

Example of the desired output format:
{
  "0": "Student Life",
  "1": "Extracurriculars",
  "2": "Academics",
  "3": "Admissions"
}

The output should be in a JSON-like format within the curly braces.
"""
)

llm = Gemini()
themes = llm(theme_prompt.format())
themes_json = json.loads(themes)
```

---

## **2. Question Template Generation**

**Prompt Template:** (as provided)

**Langchain Chain:**

```python
question_prompt_template = PromptTemplate(
    input_variables=["YOUR_THEME"],
    template="""
You are helping to generate lexicon-rich question templates for prospective students exploring the Singapore University of Technology and Design (SUTD). ...

Theme: {YOUR_THEME}  
{{
  "{{SUB_THEME_1_KEY}}": "{{SUB_THEME_1_QUERY}}",
  "{{SUB_THEME_2_KEY}}": "{{SUB_THEME_2_QUERY}}",
  "{{SUB_THEME_3_KEY}}": "{{SUB_THEME_3_QUERY}}",
  "{{SUB_THEME_4_KEY}}": "{{SUB_THEME_4_QUERY}}"
}}
"""
)

question_templates = {}
for theme in themes_json.values():
    response = llm(question_prompt_template.format(YOUR_THEME=theme))
    question_templates[theme] = json.loads(response)
```

---

## **3. Search Query Generation**

**Prompt Template:** (as provided)

**Langchain Chain:**

```python
search_query_prompt_template = PromptTemplate(
    input_variables=["QUESTION_TEMPLATE_TO_SEARCH", "LEXICON_PLACEHOLDER_TO_FILL"],
    template="""
You are an expert in generating effective web search queries for finding information related to the Singapore University of Technology and Design (SUTD). ...

Question Template: "{QUESTION_TEMPLATE_TO_SEARCH}"
LEXICON_PLACEHOLDER: {LEXICON_PLACEHOLDER_TO_FILL}
Search Query:
"""
)

search_queries = {}
for theme, subthemes in question_templates.items():
    for subtheme_key, q_template in subthemes.items():
        # Extract the placeholder, e.g., {CLUB}
        placeholder = re.search(r"\{(\w+)\}", q_template).group(0)
        prompt = search_query_prompt_template.format(
            QUESTION_TEMPLATE_TO_SEARCH=q_template,
            LEXICON_PLACEHOLDER_TO_FILL=placeholder
        )
        search_query = llm(prompt)
        search_queries[q_template] = search_query.strip()
```

---

## **4. Web Search (Tool)**

Use a web search tool (e.g., SerpAPI, Bing API, or a custom Google Search wrapper):

```python
from langchain.tools import SerpAPIWrapper

search_tool = SerpAPIWrapper()
retrieved_contexts = {}
for q_template, search_query in search_queries.items():
    results = search_tool.run(search_query)
    retrieved_contexts[q_template] = results
```

---

## **5. Lexicon Extraction**

**Prompt Template:** (as provided)

**Langchain Chain:**

```python
lexicon_extraction_prompt_template = PromptTemplate(
    input_variables=["QUESTION_TEMPLATE_TO_PARSE", "LEXICON_PLACEHOLDER_TO_EXTRACT", "YOUR_RETRIEVED_CONTEXT"],
    template="""
You are an expert in information extraction. Your goal is to parse the provided text and identify specific values that can fill the placeholder {LEXICON_PLACEHOLDER_TO_EXTRACT} in the following question template:

"{QUESTION_TEMPLATE_TO_PARSE}"

The following is the text retrieved from a web search (or other source) that may contain the information you need:

--- START OF CONTEXT ---
{YOUR_RETRIEVED_CONTEXT}
--- END OF CONTEXT ---

Based on this context, please extract all relevant and distinct values that can be used to replace the placeholder {LEXICON_PLACEHOLDER_TO_EXTRACT} in the question template. List each potential value on a new line. If no relevant values are found in the context, please state "No relevant values found."
"""
)

lexicon_values = {}
for q_template, context in retrieved_contexts.items():
    placeholder = re.search(r"\{(\w+)\}", q_template).group(0)
    prompt = lexicon_extraction_prompt_template.format(
        QUESTION_TEMPLATE_TO_PARSE=q_template,
        LEXICON_PLACEHOLDER_TO_EXTRACT=placeholder,
        YOUR_RETRIEVED_CONTEXT=context
    )
    values = llm(prompt)
    lexicon_values[q_template] = [v.strip() for v in values.split('\n') if v.strip() and "No relevant values found" not in v]
```

---

## **6. Grounded QA Generation**

**Prompt Template:** (as provided)

**Langchain Chain:**

```python
qa_generation_prompt_template = PromptTemplate(
    input_variables=["QUESTION_STRING", "RETRIEVED_CONTEXT"],
    template="""
You are a university assistant for Singapore University of Technology and Design (SUTD). Using the question below and a retrieved context, generate a concise and informative answer appropriate for a prospective student to SUTD.

Question: "{QUESTION_STRING}"

Context:
--- START OF CONTEXT ---
{RETRIEVED_CONTEXT}
--- END OF CONTEXT ---

Answer:
"""
)

qa_pairs = []
for q_template, values in lexicon_values.items():
    for value in values:
        question = q_template.replace(re.search(r"\{(\w+)\}", q_template).group(0), value)
        context = retrieved_contexts[q_template]
        prompt = qa_generation_prompt_template.format(
            QUESTION_STRING=question,
            RETRIEVED_CONTEXT=context
        )
        answer = llm(prompt)
        qa_pairs.append({
            "question": question,
            "answer": answer.strip(),
            "context": context
        })
```

---

## **7. Save as JSONL**

```python
import json

with open('sutd_qa_dataset.jsonl', 'w') as f:
    for qa in qa_pairs:
        f.write(json.dumps(qa) + '\n')
```

---

## **Summary Table of Workflow Steps**

| Step                | Input                    | Output                    | Tool/Module      |
|---------------------|-------------------------|---------------------------|------------------|
| Theme Generation    | None                    | List of themes            | LLM (Gemini)     |
| Question Templates  | Theme                   | Sub-themes & templates    | LLM (Gemini)     |
| Search Query Gen    | Q-template, Placeholder | Search query string       | LLM (Gemini)     |
| Web Search          | Search query            | Retrieved context         | Search API       |
| Lexicon Extraction  | Q-template, Context     | Lexicon values            | LLM (Gemini)     |
| QA Generation       | Q, Context              | Answer                    | LLM (Gemini)     |
| Save                | QA pairs                | JSONL file                | Python I/O       |

---

**Tips:**
- Modularize each step as a function or Langchain chain.
- Use caching or saving intermediate results to avoid redundant API calls.
- Carefully handle rate limits and API quotas for LLM and search tools.
- You may want to parallelize web search and QA generation for efficiency.

Let me know if you need a full working script or further details on any step!

---
Answer from Perplexity: pplx.ai/share