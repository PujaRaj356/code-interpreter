
import os
import sys
import io
import json
import traceback
import contextlib
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Code Interpreter API")

# AI Pipe configuration
API_KEY = os.getenv("OPENAI_API_KEY")

client = (
    OpenAI(
        api_key=API_KEY,
        base_url="https://aipipe.org/openai/v1"
    )
    if API_KEY
    else None
)

MODEL = "gpt-4.1-nano"


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


def execute_python_code(code: str) -> dict:
    """Execute code and capture its output or traceback."""
    output_buffer = io.StringIO()

    try:
        compiled_code = compile(code, "<user_code>", "exec")

        with contextlib.redirect_stdout(output_buffer):
            with contextlib.redirect_stderr(output_buffer):
                exec(compiled_code, {"__name__": "__main__"})

        return {
            "success": True,
            "output": output_buffer.getvalue()
        }

    except Exception:
        error_text = traceback.format_exc()

        return {
            "success": False,
            "output": output_buffer.getvalue() + error_text
        }


def find_error_lines(code: str, error_text: str) -> List[int]:
    """Ask AI Pipe to identify user-code line numbers in a traceback."""
    if not client:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on Render."
        )

    prompt = f"""
You are a Python traceback analyzer.

Identify the exact 1-based line numbers in the USER CODE that caused
the exception. Return only valid JSON in this format:
{{"error": [3]}}

Rules:
- Use line numbers from the provided user code, starting at 1.
- Do not count lines from Python's internal libraries or this server.
- If the traceback points to a line in the user code, report that line.
- Return an empty list only if no user-code line can be identified.
- Do not include explanations or Markdown.

USER CODE:
{code}

TRACEBACK:
{error_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Analyze Python tracebacks and return only "
                        "the requested JSON object."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content or ""
        content = content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(content)
        lines = parsed.get("error", [])

        code_line_count = len(code.splitlines())

        return sorted({
            int(line)
            for line in lines
            if isinstance(line, int)
            and 1 <= line <= code_line_count
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI error analysis failed: {str(exc)}"
        )


@app.get("/")
def home():
    return {"message": "Code Interpreter API is running"}


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    code = request.code
    execution = execute_python_code(code)

    # Successful code: return output without calling AI.
    if execution["success"]:
        return CodeResponse(
            error=[],
            result=execution["output"]
        )

    # Call AI only when execution fails.
    error_lines = find_error_lines(
        code,
        execution["output"]
    )

    return CodeResponse(
        error=error_lines,
        result=execution["output"]
    )
