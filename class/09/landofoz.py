import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
#=================================
# demonstrator for Markov Chain 
# transition probability
#=================================

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("weather",type=str,
                        help="current weather in Land of Oz:\n"
                             "   R        : rain\n"
                             "   N        : nice\n"
                             "   S        : snow")
    parser.add_argument("M",type=int,
                        help="provide forecast M days ahead")

    args       = parser.parse_args()
    weather    = args.weather
    M          = args.M
    if (M < 1):
        parser.error("Need M > 0")
    if (weather == 'R'):
        today=np.array([1.0,0.0,0.0])
    elif(weather == 'N'):
        today=np.array([0.0,1.0,0.0])
    elif(weather == 'S'):
        today=np.array([0.0,0.0,1.0])
    else: 
        raise Exception("They don't have that kind of weather in the Land of Oz: %s" % (weather))

    P = np.array([[0.50,0.25,0.25],
                  [0.50,0.00,0.50],
                  [0.25,0.25,0.50]]) 

    for i in range(M):
        forecast = np.dot(np.transpose(P),today)
        today    = forecast
    
    print("The official LOWS (Land of Oz Weather Service) forecast for %5i days from now: R=%4.2f N=%4.2f S=%4.2f\n" % (M,forecast[0],forecast[1],forecast[2]))

#=================================
main()

    


