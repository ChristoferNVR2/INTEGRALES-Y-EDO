import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from prettytable import PrettyTable


def trapezoidal(x_0, x_1, f):
    h = x_1 - x_0
    approx = (h / 2) * (f(x_0) + f(x_1))
    return approx


def error_trapezoidal(x_0, x_1, f_2_prime, num_points=100):
    h = x_1 - x_0
    x_values = np.linspace(x_0, x_1, num_points)
    second_derivative_values = [abs(f_2_prime(x)) for x in x_values]
    max_second_derivative = max(second_derivative_values)
    error = abs((h ** 3 / 12) * max_second_derivative)
    return error


def table_trapezoidal(x_0, x_1, f, f_2_prime):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = trapezoidal(x_0, x_1, f)
    error = error_trapezoidal(x_0, x_1, f_2_prime)
    table.add_row([x_0, x_1, approx, error])

    print(table)

    x_min = x_0 - 0.1
    x_max = x_1 + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_2_prime(x) for x in xAxis], label="f''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Second derivative")
    plt.xlabel("x")
    plt.ylabel("f''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def simpson(x_0, x_2, f):
    x_1 = (x_0 + x_2) / 2
    h = (x_2 - x_0) / 2
    approx = (h / 3) * (f(x_0) + 4 * f(x_1) + f(x_2))
    return approx


def error_simpson(x_0, x_2, f_4_prime, num_points=100):
    h = (x_2 - x_0) / 2
    x_values = np.linspace(x_0, x_2, num_points)
    fourth_derivative_values = [f_4_prime(x) for x in x_values]
    max_fourth_derivative = max(fourth_derivative_values)
    error = abs((h ** 5 / 90) * max_fourth_derivative)
    return error


def table_simpson(x_0, x_2, f, f_4_prime):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = simpson(x_0, x_2, f)
    error = error_simpson(x_0, x_2, f_4_prime)
    table.add_row([x_0, x_2, approx, error])

    print(table)

    x_min = x_0 - 0.1
    x_max = x_2 + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [abs(f_4_prime(x)) for x in xAxis], label="f''''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Fourth derivative")
    plt.xlabel("x")
    plt.ylabel("f''''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def simpson_three_eighths(x_0, x_3, f):
    h = (x_3 - x_0) / 3
    x_1 = x_0 + h
    x_2 = x_0 + 2 * h
    approx = (h * 3 / 8) * (f(x_0) + 3 * f(x_1) + 3 * f(x_2) + f(x_3))
    return approx


def error_simpson_three_eighths(x_0, x_3, f_4_prime, num_points=100):
    h = (x_3 - x_0) / 3
    x_values = np.linspace(x_0, x_3, num_points)
    fourth_derivative_values = [abs(f_4_prime(x)) for x in x_values]
    max_fourth_derivative = max(fourth_derivative_values)
    error = abs((h ** 5 * (3 / 80)) * max_fourth_derivative)
    return error


def table_simpson_three_eighths(x_0, x_3, f, f_4_prime):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = simpson(x_0, x_3, f)
    error = error_simpson(x_0, x_3, f_4_prime)
    table.add_row([x_0, x_3, approx, error])

    print(table)

    x_min = x_0 - 0.1
    x_max = x_3 + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_4_prime(x) for x in xAxis], label="f''''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Fourth derivative")
    plt.xlabel("x")
    plt.ylabel("f''''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def closed_newton_four(x_0, x_4, f):
    h = (x_4 - x_0) / 4
    x_1 = x_0 + h
    x_2 = x_0 + 2 * h
    x_3 = x_0 + 3 * h
    approx = (h * 2 / 45) * (7 * f(x_0) + 32 * f(x_1) + 12 * f(x_2) + + 32 * f(x_3) + 7 * f(x_4))
    return approx


def error_closed_newton_four(x_0, x_4, f_6_prime, num_points=100):
    h = (x_4 - x_0) / 4
    x_values = np.linspace(x_0, x_4, num_points)
    sixth_derivative_values = [abs(f_6_prime(x)) for x in x_values]
    max_sixth_derivative = max(sixth_derivative_values)
    error = abs((h ** 7 * (8 / 945)) * max_sixth_derivative)
    return error


def table_closed_newton_four(x_0, x_4, f, f_6_prime):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = simpson(x_0, x_4, f)
    error = error_simpson(x_0, x_4, f_6_prime)
    table.add_row([x_0, x_4, approx, error])

    print(table)

    x_min = x_0 - 0.1
    x_max = x_4 + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_6_prime(x) for x in xAxis], label="f''''''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Sixth derivative")
    plt.xlabel("x")
    plt.ylabel("f''''''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


# Open methods


def open_newton_zero(x__1, x_1, f):
    h = (x_1 - x__1) / 2
    x_0 = (x__1 + x_1) / 2
    approx = 2 * h * f(x_0)
    return approx


def error_open_newton_zero(x__1, x_1, f_2_prime, num_points=100):
    h = (x_1 - x__1) / 2
    x_values = np.linspace(x__1, x_1, num_points)
    second_derivative_values = [abs(f_2_prime(x)) for x in x_values]
    max_second_derivative = max(second_derivative_values)
    error = abs((h ** 3 / 3) * max_second_derivative)
    return error


def table_open_newton_zero(x__1, x_1, f, f_2_prime):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = open_newton_zero(x__1, x_1, f)
    error = error_open_newton_zero(x__1, x_1, f_2_prime)
    table.add_row([x__1, x_1, approx, error])

    print(table)

    x_min = x__1 - 0.1
    x_max = x_1 + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_2_prime(x) for x in xAxis], label="f''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Second derivative")
    plt.xlabel("x")
    plt.ylabel("f''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def open_newton_one(x__1, x_2, f):
    h = (x_2 - x__1) / 3
    x_0 = x__1 + h
    x_1 = x__1 + 2 * h
    approx = (h * 3 / 2) * (f(x_0) + f(x_1))
    return approx


def error_open_newton_one(x__1, x_2, f_2_prime, num_points=100):
    h = (x_2 - x__1) / 3
    x_values = np.linspace(x__1, x_2, num_points)
    second_derivative_values = [abs(f_2_prime(x)) for x in x_values]
    max_second_derivative = max(second_derivative_values)
    error = abs((h ** 3 * (3 / 4)) * max_second_derivative)
    return error


def table_open_newton_one(x__1, x_2, f, f_2_prime):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = open_newton_one(x__1, x_2, f)
    error = error_open_newton_one(x__1, x_2, f_2_prime)
    table.add_row([x__1, x_2, approx, error])

    print(table)

    x_min = x__1 - 0.1
    x_max = x_2 + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_2_prime(x) for x in xAxis], label="f''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Second derivative")
    plt.xlabel("x")
    plt.ylabel("f''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def open_newton_two(x__1, x_3, f):
    h = (x_3 - x__1) / 4
    x_0 = x__1 + h
    x_1 = x__1 + 2 * h
    x_2 = x__1 + 3 * h
    approx = (h * 4 / 3) * (2 * f(x_0) - f(x_1) + 2 * f(x_2))
    return approx


def error_open_newton_two(x__1, x_3, f_4_prime, num_points=100):
    h = (x_3 - x__1) / 4
    x_values = np.linspace(x__1, x_3, num_points)
    fourth_derivative_values = [abs(f_4_prime(x)) for x in x_values]
    max_fourth_derivative = max(fourth_derivative_values)
    error = abs((h ** 5 * (14 / 45)) * max_fourth_derivative)
    return error


def table_open_newton_two(x__1, x_3, f, f_4_prime):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = open_newton_two(x__1, x_3, f)
    error = error_open_newton_two(x__1, x_3, f_4_prime)
    table.add_row([x__1, x_3, approx, error])

    print(table)

    x_min = x__1 - 0.1
    x_max = x_3 + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_4_prime(x) for x in xAxis], label="f''''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Fourth derivative")
    plt.xlabel("x")
    plt.ylabel("f''''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def open_newton_three(x__1, x_4, f):
    h = (x_4 - x__1) / 5
    x_0 = x__1 + h
    x_1 = x__1 + 2 * h
    x_2 = x__1 + 3 * h
    x_3 = x__1 + 4 * h
    approx = (h * 5 / 24) * (11 * f(x_0) + f(x_1) + f(x_2) + 11 * f(x_3))
    return approx


def error_open_newton_three(x__1, x_4, f_4_prime, num_points=100):
    h = (x_4 - x__1) / 5
    x_values = np.linspace(x__1, x_4, num_points)
    fourth_derivative_values = [abs(f_4_prime(x)) for x in x_values]
    max_fourth_derivative = max(fourth_derivative_values)
    error = abs((h ** 5 * (95 / 144)) * max_fourth_derivative)
    return error


def table_open_newton_three(x__1, x_4, f, f_4_prime):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = open_newton_three(x__1, x_4, f)
    error = error_open_newton_three(x__1, x_4, f_4_prime)
    table.add_row([x__1, x_4, approx, error])

    print(table)

    x_min = x__1 - 0.1
    x_max = x_4 + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_4_prime(x) for x in xAxis], label="f''''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Fourth derivative")
    plt.xlabel("x")
    plt.ylabel("f''''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def table_all_simple_methods(a, b, f, f_2_prime, f_4_prime, f_6_prime):
    table = PrettyTable()
    table.field_names = ["Method", "a", "b", "Approx", "Error"]

    approx = trapezoidal(a, b, f)
    error = error_trapezoidal(a, b, f_2_prime)
    table.add_row(["Trapezoidal", a, b, approx, error])

    approx = simpson(a, b, f)
    error = error_simpson(a, b, f_4_prime)
    table.add_row(["Simpson", a, b, approx, error])

    approx = simpson_three_eighths(a, b, f)
    error = error_simpson_three_eighths(a, b, f_4_prime)
    table.add_row(["Simpson's 3/8 Rule", a, b, approx, error])

    approx = closed_newton_four(a, b, f)
    error = error_closed_newton_four(a, b, f_6_prime)
    table.add_row(["Closed Newton-Cotes (n=4)", a, b, approx, error])

    approx = open_newton_zero(a, b, f)
    error = error_open_newton_zero(a, b, f_2_prime)
    table.add_row(["Open Newton-Cotes (n=0)", a, b, approx, error])

    approx = open_newton_one(a, b, f)
    error = error_open_newton_one(a, b, f_2_prime)
    table.add_row(["Open Newton-Cotes (n=1)", a, b, approx, error])

    approx = open_newton_two(a, b, f)
    error = error_open_newton_two(a, b, f_4_prime)
    table.add_row(["Open Newton-Cotes (n=2)", a, b, approx, error])

    approx = open_newton_three(a, b, f)
    error = error_open_newton_three(a, b, f_4_prime)
    table.add_row(["Open Newton-Cotes (n=3)", a, b, approx, error])

    print(table)


# Composite methods

def composite_trapezoidal(a, b, f, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    approx = f(a) + f(b)

    for i in range(1, n):
        approx += 2 * f(x[i])

    approx *= h / 2
    return approx


def error_composite_trapezoidal(a, b, f_2_prime, n, num_points=100):
    h = (b - a) / n
    x_values = np.linspace(a, b, num_points)
    second_derivative_values = [abs(f_2_prime(x)) for x in x_values]
    max_second_derivative = max(second_derivative_values)
    error = abs((1 / 12) * ((b - a) * (h ** 2)) * max_second_derivative)
    return error


def table_composite_trapezoidal(a, b, f, f_2_prime, n):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = composite_trapezoidal(a, b, f, n)
    error = error_composite_trapezoidal(a, b, f_2_prime, n)
    table.add_row([a, b, approx, error])

    print(table)

    x_min = a - 0.1
    x_max = b + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_2_prime(x) for x in xAxis], label="f''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Second derivative")
    plt.xlabel("x")
    plt.ylabel("f''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def composite_simpsom(a, b, f, n):
    if n % 2 != 0:
        print("n must be an even number")
        return None

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    approx = f(a) + f(b)

    for i in range(1, n):
        if i % 2 == 0:
            approx += 2 * f(x[i])
        else:
            approx += 4 * f(x[i])

    approx *= h / 3
    return approx


def error_composite_simpsom(a, b, f_4_prime, n, num_points=100):
    if n % 2 != 0:
        print("n must be an even number")
        return None

    h = (b - a) / n
    x_values = np.linspace(a, b, num_points)
    fourth_derivative_values = [abs(f_4_prime(x)) for x in x_values]
    max_fourth_derivative = max(fourth_derivative_values)
    error = abs((1 / 180) * ((b - a) * (h ** 4)) * max_fourth_derivative)
    return error


def table_composite_simpsom(a, b, f, f_4_prime, n):
    table = PrettyTable()
    table.field_names = ["a", "b", "Approx", "Error"]

    approx = composite_simpsom(a, b, f, n)
    error = error_composite_simpsom(a, b, f_4_prime, n)
    table.add_row([a, b, approx, error])

    print(table)

    x_min = a - 0.1
    x_max = b + 0.1
    xAxis = np.linspace(x_min, x_max, 100)
    plt.plot(xAxis, [f_4_prime(x) for x in xAxis], label="f''''(x)")
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title("Fourth derivative")
    plt.xlabel("x")
    plt.ylabel("f''''(x)")
    plt.grid(True, which='both')
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()


def table_all_composite_methods(a, b, f, f_2_prime, f_4_prime, n):
    table = PrettyTable()
    table.field_names = ["Method", "a", "b", "Approx", "Error"]

    approx = composite_trapezoidal(a, b, f, n)
    error = error_composite_trapezoidal(a, b, f_2_prime, n)
    table.add_row(["Composite Trapezoidal Rule", a, b, approx, error])

    approx = composite_simpsom(a, b, f, n)
    error = error_composite_simpsom(a, b, f_4_prime, n)
    table.add_row(["Composite Simpson's Rule", a, b, approx, error])

    print(table)


def euler(f, equation_name, a, b, y0, n):
    h = (b - a) / n
    x = a
    y = y0
    x_values = [x]
    y_values = [y]
    fxy = f(x, y)
    t = PrettyTable(['n', 't', f'y(t) = y(t_i-1) + h * f(t_i-1, y_i-1)', f'f(t, y) = {equation_name}'])
    t.add_row([0, x, y, fxy])

    for i in range(1, n + 1):
        y = y + h * f(x, y)
        x = a + i * h
        x_values.append(x)
        y_values.append(y)

        if i != n:
            fxy = f(x, y)
        else:
            fxy = ' '

        t.add_row([i, round(x, 4), y, fxy])

    print(t)

    plt.plot(x_values, y_values, '-o', label='y(t)', markersize=5)
    plt.axhline(color='black', linewidth=0.5)
    plt.axvline(color='black', linewidth=0.5)

    plt.title('Euler Method')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid(True, which='both')
    plt.ylim(min(y_values) - 0.1, max(y_values) + 0.1)
    plt.legend()
    plt.show()

    return y


def get_input(prompt):
    while True:
        value = input(prompt)
        if 'pi' in value:
            try:
                k = float(value.split('*')[0].strip())
                return k * np.pi
            except ValueError:
                print("Invalid input. Please enter a numeric value or a multiple of pi (e.g., '0.5', '2*pi').")
        else:
            try:
                return float(value)
            except ValueError:
                print("Invalid input. Please enter a numeric value or a multiple of pi (e.g., '0.5', '2*pi').")


def main():
    x = symbols('x')

    while True:
        print("\nSelect the procedure you'd like to do: ")
        print("1. Approximate the integral of a function")
        print("2. Approximate an differential equation")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':

            equation_name = input('\nEnter the function: ')
            try:
                func = sympify(equation_name)
                fn = func
            except Exception as e:
                print("Error defining the function: ", e)
                exit()

            f = lambdify(x, fn)
            f_2_prime = lambdify(x, diff(fn, x, x))
            f_4_prime = lambdify(x, diff(fn, x, x, x, x))
            f_6_prime = lambdify(x, diff(fn, x, x, x, x, x, x))

            print("Input the limits of integration for the function f(x):")

            a = get_input("Enter the lower limit a: ")
            b = get_input("Enter the upper limit b: ")

            while True:
                print("\nSelect a type of method to approximate the integral:")
                print("1. Simple")
                print("2. Compose")
                print("3. Exit")

                choice2 = input("Enter your choice: ")

                if choice2 == '1':
                    while True:
                        print("\nSelect the type of simple method:")
                        print("1. Closed")
                        print("2. Open")
                        print("3. All simple methods")

                        choice3 = input("Enter your choice: ")

                        if choice3 in ['1', '2', '3']:
                            break
                        else:
                            print("Invalid choice. Please enter a valid option.")

                    if choice3 == '1':
                        while True:
                            print("\nSelect a method to approximate the integral:")
                            print("1. Trapezoidal")
                            print("2. Simpson")
                            print("3. Simpson's 3/8 Rule")
                            print("4. Closed Newton-Cotes (n=4)")

                            choice4 = input("Enter your choice: ")

                            if choice4 in ['1', '2', '3', '4']:
                                break
                            else:
                                print("Invalid choice. Please enter a valid option.")

                        if choice4 == '1':
                            table_trapezoidal(a, b, f, f_2_prime)
                        elif choice4 == '2':
                            table_simpson(a, b, f, f_4_prime)
                        elif choice4 == '3':
                            table_simpson_three_eighths(a, b, f, f_4_prime)
                        elif choice4 == '4':
                            table_closed_newton_four(a, b, f, f_6_prime)

                    elif choice3 == '2':
                        while True:
                            print("\nSelect a method to approximate the integral:")
                            print("1. Open Newton-Cotes (n=0)")
                            print("2. Open Newton-Cotes (n=1)")
                            print("3. Open Newton-Cotes (n=2)")
                            print("4. Open Newton-Cotes (n=3)")

                            choice4 = input("Enter your choice: ")

                            if choice4 in ['1', '2', '3', '4']:
                                break
                            else:
                                print("Invalid choice. Please enter a valid option.")

                        if choice4 == '1':
                            table_open_newton_zero(a, b, f, f_2_prime)
                        elif choice4 == '2':
                            table_open_newton_one(a, b, f, f_2_prime)
                        elif choice4 == '3':
                            table_open_newton_two(a, b, f, f_4_prime)
                        elif choice4 == '4':
                            table_open_newton_three(a, b, f, f_4_prime)

                    elif choice3 == '3':
                        table_all_simple_methods(a, b, f, f_2_prime, f_4_prime, f_6_prime)

                elif choice2 == '2':
                    while True:
                        try:
                            n = int(input("Enter the number of sub-intervals n (even number): "))
                            if n % 2 == 0:
                                break
                            else:
                                print("Please enter an even number for the number of sub-intervals.")
                        except ValueError:
                            print("Please enter a valid integer for the number of sub-intervals.")

                    while True:
                        print("\nSelect the type of composite method:")
                        print("1. Composite Trapezoidal Rule")
                        print("2. Composite Simpson's Rule")
                        print("3. All compose methods")

                        choice3 = input("Enter your choice: ")

                        if choice3 in ['1', '2', '3']:
                            break
                        else:
                            print("Invalid choice. Please enter a valid option.")

                    if choice3 == '1':
                        table_composite_trapezoidal(a, b, f, f_2_prime, n)
                    if choice3 == '2':
                        table_composite_simpsom(a, b, f, f_4_prime, n)
                    if choice3 == '3':
                        table_all_composite_methods(a, b, f, f_2_prime, f_4_prime, n)

                elif choice2 == '3':
                    print("Exiting program...")
                    break
                else:
                    print("Invalid choice. Please enter a valid option.")
                    continue
        elif choice == '2':
            t = Symbol('t')
            y = Symbol('y')
            equation_name = input("\nEnter the function of the form y' = f(t, y): ")
            try:
                func = sympify(equation_name)
                fn = func
            except Exception as e:
                print("Error defining the function: ", e)
                exit()

            f = lambdify((t, y), fn, 'numpy')

            a = get_input('Enter the initial value of t: ')
            b = get_input('Enter the final value of t: ')
            y0 = get_input('Enter the initial value of y: ')
            n = int(input('Enter the number of iterations: '))
            euler(f, equation_name, a, b, y0, n)
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")
            continue


if __name__ == '__main__':
    main()
