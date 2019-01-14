#Shekhar Dewan
import numpy as np

def function1(numExamples = 300,sd=5):
    x_vals = []
    #print("NumExamples ->",numExamples)
    for i in range(numExamples):
        x_vals.append(generation_function())
    final_array = np.empty([numExamples,3])
    y = np.empty([numExamples])
    for i in range(numExamples):
        final_array[i][0] = x_vals[i]**2
        final_array[i][1] = x_vals[i]
        final_array[i][2] = 1
        y[i] = 0.5*final_array[i][0] + 3*final_array[i][1] + -20*final_array[i][2]
        noise = noise_gen(sd)
        y[i] += noise
    return final_array, y

def function2(numExamples = 300,sd=5):
    x_vals = []
    for i in range(numExamples):
        x_vals.append(generation_function())
    final_array = np.empty([numExamples,3])
    y = np.empty([numExamples])
    for i in range(numExamples):
        final_array[i][0] = x_vals[i]**4
        final_array[i][1] = x_vals[i]**2
        final_array[i][2] = 1
        y[i] = 0.01*final_array[i][0] + -0.2*final_array[i][1] + 2*final_array[i][2]
        noise = noise_gen(sd)
        y[i] += noise
    return final_array, y

def function3(numExamples = 300,sd=5):
    x_vals = np.empty([numExamples,4])
    for i in range(numExamples):
        x_vals[i][0] = (generation_function())
        x_vals[i][1] = (generation_function())
        x_vals[i][2] = (generation_function())
        x_vals[i][3] = (generation_function())
    final_array = np.empty([numExamples,5])
    y = np.empty([numExamples])
    #coefficients = array[3,5,-6,1,-1]
    #powers = array[1,1,1,1,0]
    for i in range(numExamples):
        final_array[i][0] = x_vals[i][0]
        final_array[i][1] = x_vals[i][1]
        final_array[i][2] = x_vals[i][2]
        final_array[i][3] = x_vals[i][3]
        final_array[i][4] = 1
        y[i] = 3*final_array[i][0] + 5*final_array[i][1] + -6*final_array[i][2] + \
                            final_array[i][3] + -1*final_array[i][4]
        noise = noise_gen(sd)
        y[i] += noise
    return final_array, y
    #x1,x2,-x3,x4

def generation_function():
    x = np.random.uniform(-10,10,1)
    return x

def regression(dataset,y):
    w = np.linalg.inv(dataset.T @ dataset) @ dataset.T @ y
    #print(w)
    return w

def train_and_test(numExamples=100,sd=5, test_size=50):
    if(numExamples <= 100):
        test_size = 50
    else:
        test_size = numExamples//4
    dataset1, y1 = function1(numExamples,sd)
    dataset2, y2 = function2(numExamples,sd)
    dataset3, y3 = function3(numExamples,sd)
    coefficients1 = regression(dataset1,y1)
    return test(coefficients1, test_size, 1)
    coefficients2 = regression(dataset2,y2)
    return test(coefficients2, test_size, 2)
    coefficients3 = regression(dataset3,y3)
    return test(coefficients3, test_size, 3)
    #regression(dataset2,y2)
    #regression(dataset3,y3)


def function1_test(x_vals, coefficients):
    numTests = len(x_vals)
    final_array = np.empty([numTests,3])
    y = np.empty([numTests])
    for i in range(numTests):
        final_array[i][0] = x_vals[i][0]**2
        final_array[i][1] = x_vals[i][0]
        final_array[i][2] = 1
        y[i] = coefficients[0]*final_array[i][0] + coefficients[1]*final_array[i][1] + \
               coefficients[2]*final_array[i][2]
    return y


def function2_test(x_vals, coefficients):
    numTests = len(x_vals)
    final_array = np.empty([numTests,3])
    y = np.empty([numTests])
    for i in range(numTests):
        final_array[i][0] = x_vals[i][0]**4
        final_array[i][1] = x_vals[i][0]**2
        final_array[i][2] = 1
        y[i] = coefficients[0]*final_array[i][0] + coefficients[1]*final_array[i][1] + \
               coefficients[2]*final_array[i][2]
    return y

def function3_test(x_vals, coefficients):
    numTests = len(x_vals)
    final_array = np.empty([numTests,5])
    y = np.empty([numTests])
    for i in range(numTests):
        final_array[i][0] = x_vals[i][0]
        final_array[i][1] = x_vals[i][1]
        final_array[i][2] = x_vals[i][2]
        final_array[i][3] = x_vals[i][3]
        final_array[i][4] = 1
        y[i] = coefficients[0]*final_array[i][0] + coefficients[1]*final_array[i][1] + \
               coefficients[2]*final_array[i][2] + \
               coefficients[3]*final_array[i][3] + coefficients[4]*final_array[i][4]
    return y

def test(coefficients, numTests=50, test_vect=1):
    x_vals = np.empty([numTests,4])
    coefficientsf1 = [0.5,3,-20]
    coefficientsf2 = [0.01,-0.2,2]
    coefficientsf3 = [3,5,-6,1,-1]
    for i in range(numTests):
        x_vals[i][0] = (generation_function())
        x_vals[i][1] = (generation_function())
        x_vals[i][2] = (generation_function())
        x_vals[i][3] = (generation_function())
    ytrain = []
    yoriginal = []
    if(test_vect == 1):
        ytrain = function1_test(x_vals, coefficients)
        yoriginal = function1_test(x_vals, coefficientsf1)
        #print("for the 1st function:")
    elif(test_vect == 2):
        ytrain = function2_test(x_vals, coefficients)
        yoriginal = function2_test(x_vals, coefficientsf2)
        #print("for the 2nd function:")
    elif(test_vect == 3):
        ytrain = function3_test(x_vals, coefficients)
        yoriginal = function3_test(x_vals, coefficientsf3)
        #print("for the 3rd function:")
    difference = []
    for i in range(len(yoriginal)):
        difference.append(ytrain[i] - yoriginal[i])
    return difference
    #print("trained-original(avg) = ", np.mean(difference),"sd =", np.std(difference))
    #generate more values
    #compare y's

def noise_gen(sd):
    return np.random.normal(0,sd,1)

def test_many_times(low, high, increment, sd = False,reps = 10,examples = 1000):
    avg_diff = []
    avg_sd = []
    diff_at = []
    sd_at = []
    for i in range(low,high,increment):
        for j in range (reps):
            if(sd):
                difference = train_and_test(examples,i)
            else:
                difference = train_and_test(i,5)
            avg_diff.append(np.mean(difference))
            avg_sd.append(np.std(difference))
        if(sd):
            print("sd is",i, ",Examples for training are", examples, "testing are", examples//4)
        else:
            if(i <= 100):
                print("sd is 5", ",Examples for training are",i,"testing are",50)
            else:
                print("sd is 5", ",Examples for training are",i,"testing are",i//4)
        print("over", reps,"trials, trained-original(avg) = ", np.mean(avg_diff),"sd =",
              np.std(avg_sd))
        #sd_at.append(np.mean(avg_sd))
    #return diff_at,sd_at

def test_everything():
    avg_diff = []
    avg_sd = []
    for i in range(1,21):
        for j in range (10):
            print("sd is",i)
            print("Examples for training are 2000, testing are 500")
            difference = train_and_test(400,i)
            avg_diff.append(np.mean(difference))
            avg_sd.append()
    print("trained-original(avg) = ", np.mean(difference),"sd =", np.std(difference))
    for i in range(10,110,10):

        for j in range (10):
            print("sd is 5")
            print("Examples for training are",i,"testing are",i//4)
            train_and_test(i,5)
            difference = train_and_test(400,i)
        print("trained-original(avg) = ", np.mean(difference),"sd =", np.std(difference))
    for i in range(100,10100,500):
        print("sd is 5")
        print("Examples for training are",i,"testing are",i//4)
        train_and_test(i,5)
        difference = train_and_test(400,i)
        print("trained-original(avg) = ", np.mean(difference),"sd =", np.std(difference))


test_many_times(1,21,1, True,10,250)
test_many_times(10,110,10,False)
test_many_times(200,10100,500,False,2)