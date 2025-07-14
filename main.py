import sqlite3

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DB_URI = "reviews.db"
SENTIMENT_CLASSIFIER = {
    "positive": ["люблю", "хорошо", "отлично"],
    "negative": ["плохо", "ужасно", "отвратительно"]
}


class ReviewRequest(BaseModel):
    text: str = Field(..., description="Текст отзыва")

class ReviewResponse(ReviewRequest):
    id: int
    sentiment: str = Field(..., description="Метка настроения: 'positive' или 'negative'")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def predict(text: str) -> str:
    for key, value in SENTIMENT_CLASSIFIER.items():
        for item in value:
            if item in text:
                return key
    else:
        return "neutral"


def create_db_connection(db_uri: str) -> sqlite3.Connection:
    return sqlite3.connect(db_uri)


def write_into_db(data: ReviewRequest, label: str) -> ReviewResponse:
    conn = create_db_connection(DB_URI)
    cur = conn.cursor()
    created_at = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "INSERT INTO reviews (text, sentiment, created_at) VALUES  (?, ?, ?)",
        (data.text, label, created_at)
    )
    conn.commit()
    review_id = cur.lastrowid
    conn.close()
    return ReviewResponse(
        id=review_id,
        text=data.text,
        sentiment=label,
        created_at=created_at
    )


def read_from_db(filter_label: str | None) -> List[ReviewResponse]:
    statement_all = """
            SELECT id, text, sentiment, created_at
            FROM reviews
            ORDER BY created_at DESC
            """
    
    statement_filtered = """
            SELECT id, text, sentiment, created_at
            FROM reviews
            WHERE sentiment = ?
            ORDER BY created_at DESC
            """

    conn = create_db_connection(DB_URI)
    cur = conn.cursor()

    if filter_label:
        cur.execute(statement_filtered, (filter_label,))
    else:
        cur.execute(statement_all)

    rows = cur.fetchall()
    conn.close()
    return [
        ReviewResponse(
            id=row[0],
            text=row[1],
            sentiment=row[2],
            created_at=row[3]
        ) for row in rows
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = create_db_connection(DB_URI)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    created_at TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()
    yield
    

app = FastAPI(lifespan=lifespan)


@app.post("/reviews")
async def add_review(data: ReviewRequest) -> ReviewResponse:
    label = predict(data.text)
    response = write_into_db(data, label)
    return response


@app.get("/reviews")
async def get_reviews(sentiment: Optional[str] = None) -> List[ReviewResponse]:
    if sentiment not in ["positive", "negative", "neutral", None]:
        raise HTTPException(status_code=400, detail=f"Accepted keywords are: positive, negative or neutral. Got {sentiment}.")
    result = read_from_db(sentiment)
    if not result:
        raise HTTPException(status_code=404, detail=f"No data for key {sentiment}!")
    return result
