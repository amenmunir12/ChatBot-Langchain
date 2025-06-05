# CHAT BOT USING LANGCHAIN AND CHATGPT

##  Work Completed So Far

### 1.  Backend Setup (Django)
- Initialized Django project
- Created a Django app for document handling
- Defined and added models for file uploads
- Ran migrations to update the database schema

### 2. API Testing with Postman
- Uploaded documents using `POST` request (with `form-data`)
- Fetched uploaded documents using `GET` request
- Sent dummy user input via `POST` to test chat response
- Tested both `form-data` (for document) and `JSON` (for metadata) formats
- Confirmed API call structure for user-question handling

### 3. ChatGPT Integration 
- Set up OpenAI API key using github keys for backend usage
- Implemented and debugged API calls to ChatGPT
- Created a view to receive user input
- Successfully made OpenAI GPT calls from Django backend and have chatgpt responses to user input 

### 3. Langchain Integration 
-turning fies into chunks of 1000 with overlap of 150 

##  In progress
- currently learning how to vectorize the uploaded docs and store them in DB 
