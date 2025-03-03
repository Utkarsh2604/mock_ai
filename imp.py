from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from pydantic import BaseModel, validator
import pyttsx3
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in environment variables.")
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Set it in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2000,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Templates for HTML rendering
templates = Jinja2Templates(directory="templates")  # Create a 'templates' directory

# Data models
class TopicRequest(BaseModel):
    topic: str
    difficulty: str

    @validator('difficulty')
    def difficulty_must_be_valid(cls, value):
        if value not in ['easy', 'medium', 'hard']:
            raise ValueError('Difficulty must be one of: easy, medium, hard')
        return value

class AnswerRequest(BaseModel):
    answer: str

# Global state
questions_data: Dict[str, any] = {}  # Store interview data
user_answers: List[Dict[str, str]] = []  # Store user answers and evaluations

# Serve static files (for audio files)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper functions
def generate_interview_questions(topic, difficulty):
    """Generate interview questions based on topic and difficulty."""
    prompt = (
        f"You are an expert mock interviewer. Generate five interview questions on '{topic}' "
        f"at a '{difficulty}' difficulty level. Return them as a numbered list where each question is a new line.  Do not use any bold or markdown."
    )
    try:
        response = model.generate_content(prompt)
        logging.info(f"Token usage for question generation: {response.usage_metadata}")
        questions = response.text.strip().split("\n")
        questions = [q.split(' ', 1)[1].strip() if len(q.split(' ', 1)) > 1 else q.strip() for q in questions]
        return questions
    except Exception as e:
        logging.error(f"Error generating questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate questions.")

def text_to_speech(text, filename):
    """Convert text to speech and save as an audio file."""
    try:
        tts_engine.save_to_file(text, filename)
        tts_engine.runAndWait()
        logging.info(f"Audio file created: {filename}")
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio.")

# API endpoints
@app.post("/start_interview")
def start_interview(request: TopicRequest):
    topic = request.topic
    difficulty = request.difficulty
    try:
        questions = generate_interview_questions(topic, difficulty)

        if len(questions) < 5:
            raise HTTPException(status_code=500, detail="Failed to generate questions.")

        questions_data["questions"] = questions
        questions_data["current_index"] = 0
        questions_data["evaluations"] = []  # Initialize evaluations list
        user_answers.clear()

        if not os.path.exists("static"):
            os.makedirs("static")

        audio_filename = f"static/question_0.mp3"
        text_to_speech(questions[0], audio_filename)

        return {"question": questions[0], "audio_file": audio_filename}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/answer")
def submit_answer(request: AnswerRequest):
    try:
        if "questions" not in questions_data or questions_data["current_index"] >= len(questions_data["questions"]):
            raise HTTPException(status_code=400, detail="No active interview.")

        current_question = questions_data["questions"][questions_data["current_index"]]
        user_answers.append({"question": current_question, "answer": request.answer})

        # Evaluate the answer
        evaluation_prompt = (
            f"Evaluate the following candidate response based on correctness, depth, and relevance:\n"
            f"Q: {current_question}\nA: {request.answer}\n"
            "Provide a detailed evaluation of the candidate's performance."
        )
        evaluation_response = model.generate_content(evaluation_prompt)
        logging.info(f"Token usage for evaluation: {evaluation_response.usage_metadata}")
        evaluation_text = evaluation_response.text
        questions_data["evaluations"].append({"question": current_question, "answer": request.answer, "evaluation": evaluation_text})

        # Generate follow-up question
        follow_up_prompt = (
            f"Based on the candidate's answer, generate a follow-up question. Only give the question.\n"
            f"Q: {current_question}\nA: {request.answer}\n"
        )
        follow_up_response = model.generate_content(follow_up_prompt)
        logging.info(f"Token usage for follow-up question generation: {follow_up_response.usage_metadata}")
        next_question = follow_up_response.text.strip()

        # Move to the next stage (either next question or end of interview)
        questions_data["current_index"] += 1
        if questions_data["current_index"] >= len(questions_data["questions"]):
            # Interview is complete, return evaluations
            return {"evaluations": questions_data["evaluations"], "message": "Interview completed."}

        # Generate audio for the next question
        audio_filename = f"static/question_{questions_data['current_index']}.mp3"
        text_to_speech(next_question, audio_filename)

        return {
            "evaluation": evaluation_text,
            "next_question": next_question,
            "audio_file": audio_filename
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error submitting answer: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})  # Using Jinja2 templates

@app.get("/evaluations")
def get_evaluations():
    """Endpoint to retrieve all evaluations."""
    if "evaluations" in questions_data:
        return {"evaluations": questions_data["evaluations"]}
    else:
        return {"evaluations": [], "message": "No evaluations available."}

# Main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)