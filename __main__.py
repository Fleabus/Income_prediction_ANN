from adult_data import Adult_Data
if __name__ == '__main__':
    adult = Adult_Data()
    x, y = adult.get_data("train", 100, 10)
    print(x, y)
    print("Life goes on")
