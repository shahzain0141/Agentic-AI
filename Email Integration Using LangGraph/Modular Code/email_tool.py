import yagmail
from config import EMAIL_USER, EMAIL_PASSWORD
from langchain.tools import tool

@tool
def send_email(recipient: str, question: str, answer: str) -> str:
    """Send an email with the answer to the given question."""
    try:
        yag = yagmail.SMTP(user=EMAIL_USER, password=EMAIL_PASSWORD)
        subject = f"Response to: {question[:50]}"
        yag.send(to=recipient, subject=subject, contents=answer)
        return f"Email sent to {recipient}"
    except Exception as e:
        return f"Failed to send email: {e}"
