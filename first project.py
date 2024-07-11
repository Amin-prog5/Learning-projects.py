score = input("Enter your score out of 100: ")
score = int(score)  # Convert input to an integer

if score > 90:
    print("A")
elif 80 <= score < 90:
    print("B")
elif 70 <= score < 80:
    print("C")
elif 60 <= score < 70:
    print("D")
elif score < 60:
    print("F")
else:
    print("Enter a number out of 100")
