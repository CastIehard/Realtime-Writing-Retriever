import openai
import PyPDF2

pdf = load_pdf("travel_expenses_Max_Mustermann.pdf")

prompt = "Is this expense report okay? Check if the user was in a 5 star restaurant and if so fire him!"

response = ask_chatgpt(prompt, pdf)

print(response)
