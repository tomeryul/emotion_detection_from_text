from ensembleModel import TE_Model

model = TE_Model()


def menu():
    print(f'(1) predict emotion\n'
          f'(2) show tokenizer\n'
          f'(0) exit')


while True:
    menu()
    option = int(input("enter your option: "))
    if option == 1:
        text = str(input("enter a sentence: "))
        answer = model.res_of_two_models(text)
        model.better_printing(answer)

    elif option == 2:
        text = str(input("enter a sentence: "))
        answer = model.text_to_vector(text)
        print(answer)

    elif option == 0:
        print("come back soon!!\n")
        break

    else:
        print(f'Not valid option!\n')
