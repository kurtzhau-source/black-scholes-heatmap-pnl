import streamlit as st
import numpy as np
from scipy.stats import norm 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
from streamlit_extras.stylable_container import stylable_container



st.set_page_config(layout="wide")
                   

#set up the title
st.title("Modelling the Black-Scholes pricing equation for European stock options with a Profit & Loss heatmap for visualisation")

st.write("---")

calculate = st.button("Calculate")

st.write("---")





#creating a function to compute the price for a call
def black_scholes_call(S0, K, r, sigma, t):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    call = S0*norm.cdf(d1) - K*np.exp(-t*r)*norm.cdf(d2)
 
    return call

#creating a function to compute the price for a put
def black_scholes_put(S0, K, r, sigma, t):
    d1 = (np.log(S0/K) + (r + (sigma**2)/2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    put = K*np.exp(-t*r)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return put



#create a function to  find the partition's of our intervals for the heat map
def partition(first,last, number):
    spread = last - first
    intervals = []
    for i in range(number):
        intervals.append(round(spread*(i/(number-1)) + first,2))
    return intervals



#function that id's negative pnls and returns the cell colour as red, otherwise green
def color_cells(s):
    threshold = 0
    if s < threshold:
        return "background-color:red"
    else:
        return "background-color:green"            

#I want a function that contains my container styling in CSS so it is all centeralised and consistent
def css_stylers():
        return " padding:5px; border-radius:3px; border: .65px solid; font-weight:bold; max-width: 150px"

#create an easy function to index dataframes easily 
def index_label(matrix):
    matrix.index = spot_array
    matrix.columns = vol_array
    return matrix

#get ready for some self plugging
url = "https://www.linkedin.com/in/alexander-lingstadt-page-84064132b/"



#Collect our variables for the normal case on the sidebar
with st.sidebar: 

    st.write("Created by [Alexander Page](%s)" % url)

   

    
    strike_price = round(st.number_input("Strike Price", min_value=0.01, value=42.00, step=0.01),2)
    
    initial_price = round(st.number_input("Initial Price", min_value=0.01, value=40.00, step=0.01),2)
    
    
    interest = round(st.number_input("Risk free interest", min_value=0.01, max_value=1.00, value=0.05, step=0.01),2)
    
    volatility = round(st.number_input("Stock volatility", min_value=0.01, value=0.4, max_value=1.00, step=0.01),2)

    time = round(st.number_input("Time to maturity (years)", min_value=0.01, value=0.5, step=0.01),2) 
    
    call_price = round(st.number_input("Please enter the current call price offered", min_value=0.00, value=4.50, step=0.01),2)

    put_price = round(st.number_input("Please enter the current put price offered", min_value=0.00, value=4.50, step=0.01),2)
                                              



#then create a line to seperate the normal variables from the heatmap interval variables

    st.markdown("---")

    st.header("Heat map parameters")

    #here we chose these variables to be changeable as the black scholes equation is most sensitive to these two inputs 

    lower_vol = round(st.slider("Lower bound for stock volatility", min_value=0.01, value=0.01, max_value=0.99, step=0.01),2)

    
    upper_vol = round(st.slider("Upper bound for stock volatility", min_value=lower_vol, value=0.80, max_value=1.00, step=0.01),2)
    
    lower_spot = round(st.number_input("Please enter the lower bound of the spot price", min_value=0.00, value=35.00, step=0.01),2)

    upper_spot = round(st.number_input("Please enter the upper bound of the spot price", min_value=lower_spot +0.01, value = lower_spot*1.5, step=0.01),2)





    num = int(st.number_input("Please enter the dimension for the nxn profit and loss heat map", min_value=2, value=6, step=1))

    annot = st.radio("Do you want annotations?",
             ["Yes", "No"], captions=["", "Recommended for higher dimensions"])


#start outputting the computed variables 
bs_call_price = round(black_scholes_call(initial_price, strike_price, interest, volatility, time),2)
bs_put_price = round(black_scholes_put(initial_price, strike_price, interest, volatility, time),2)

call_base_pnl = round(bs_call_price - call_price,2)
put_base_pnl = round(bs_put_price - put_price,2) 

#############################################################################################




if calculate: 
#Displaying the price and predicted pnl given our current values
#create in columns 
    call_con, put_con = st.columns(2)


    with call_con:
        st.write("This is the predicted price of the call option.")
        with st.container(border = False):
            #here we are just using some css to style the box to our desire 
            st.markdown(f"""
            <div style = "{css_stylers()}">
           £{bs_call_price}
            </div>
            """, unsafe_allow_html=True)
        st.write("This is the predicted PnL of the call option given its actual price.")
        with st.container(border=False):
            st.markdown(f"""
            <div style = "{color_cells(call_base_pnl)}; {css_stylers()}">
            £{call_base_pnl}
            </div>
            """, unsafe_allow_html=True)




    with put_con:
        st.write("This is the predicted price of the put option.")
        with st.container(border=False):
            st.markdown(f"""
            <div style = "{css_stylers()}" >
            £{bs_put_price}
            </div>
            """, unsafe_allow_html=True)

            st.write("This is the predicted PnL of the put option given its actual price.")
            #here we use our css stylers function and the color cells function to give the conditional background
            with st.container(border=False):
                st.markdown(f"""
                <div style = "{color_cells(put_base_pnl)}; {css_stylers()}">
                £{put_base_pnl}
                </div>
                """, unsafe_allow_html=True)



    



#building the heatmap
#give some space for the heatmaps to breath
    st.markdown("---")

    st.header("Heatmaps of predicted PnL")

    with st.container():
        st.write("Below are the two PnL heatmaps, in each square the upper value represents the predicted PnL while the lower is the predicted price of the option. ")


#building the infastrucutre for the heatmap

    vol_spread = partition(lower_vol, upper_vol, num)
    spot_spread = partition(lower_spot, upper_spot, num)

#set as array's so we can use them for indexing 
    vol_array = np.array(vol_spread)
    spot_array = np.array(spot_spread)
########################################################################################

#create matrix to store heatmap call values in 
    matrix_call = [[round(black_scholes_call(spot_spread[i], strike_price, interest, vol_spread[j],time),2)  for i in range(num)] for j in range(num)]

#then create a dataframe for matrix for display purposes
    df_call = pd.DataFrame(matrix_call)
#use the labelling matrix
    df_call = index_label(df_call)

#create a matrix to store the pnl's to use as a basis for our styling
    matrix_call_pnl = [[round(matrix_call[i][j] - call_price,2) for j in range(num)] for i in range(num)]

#create dataframe
    df_call_pnl = pd.DataFrame(matrix_call_pnl)
    df_call_pnl = index_label(df_call_pnl)


#create matrix to store heatmap put values in
    matrix_put = [[round(black_scholes_put(spot_spread[i], strike_price, interest, vol_spread[j], time),2) for i in range(num)] for j in range(num)]
#then create a dataframe for matrix for display purposes
    df_put =pd.DataFrame(matrix_put)
    df_put = index_label(df_put)

#create matrix to store predicted pnl for the put values 
    matrix_put_pnl = [[round(matrix_put[i][j] - put_price,2) for j in range(num)] for i in range(num)]
#respective data frame 
    df_put_pnl = pd.DataFrame(matrix_put_pnl)
    df_put_pnl = index_label(df_put_pnl)


#################################################
#Construct the heatmaps
    
    if annot == "Yes":
     
#set up subplots to draw heatmap on
        fig1, ax = plt.subplots()


#first annotate with the predicted call price  
        sns.heatmap(df_call_pnl, annot=df_call, annot_kws={'va':'top', "color" : "black"}, fmt="", cbar=False)
#next draw the heatmap and annotate with the predicted pnl 
        sns.heatmap(df_call_pnl, annot=True, center = 0, cmap = LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256),annot_kws={'va':'bottom', "color" : "black"},fmt="", cbar=True )

        plt.title('Heatmap of predicted PnL of a Call option (£)', fontsize = 15) 
        plt.xlabel('Volatility', fontsize = 10) 
        plt.ylabel('Spot Price', fontsize = 10) 

#set up subplots to draw heatmap on
        fig2, ax = plt.subplots()



#first annotate with the predicted put price 
        sns.heatmap(df_put_pnl, annot=df_put, annot_kws={'va':'top', "color" : "black"}, fmt="", cbar=False)
#next draw the heatmap and annotate with the predicted pnl
        sns.heatmap(df_put_pnl, annot=True, center = 0, cmap = LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256),annot_kws={'va':'bottom', "color" : "black"},fmt="", cbar=True )

        plt.title("Heatmap of predicted Pnl of a Put option (£)", fontsize = 15) 
        plt.xlabel('Volatility', fontsize = 10) 
        plt.ylabel('Spot Price', fontsize = 10) 


######
#create columns to display the heatmaps in repsective and display
        col_call_map, col_put_map = st.columns(2)

        with col_call_map:
            st.pyplot(fig1)

        with col_put_map:
            st.pyplot(fig2)
    else:
        fig1, ax = plt.subplots()



#next draw the heatmap and annotate with the predicted pnl 
        sns.heatmap(df_call_pnl, annot=False, center = 0, cmap = LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256),annot_kws={'va':'bottom', "color" : "black"},fmt="", cbar=True )

        plt.title('Heatmap of predicted PnL of a Call option (£)', fontsize = 15) 
        plt.xlabel('Volatility', fontsize = 10) 
        plt.ylabel('Spot Price', fontsize = 10) 

#set up subplots to draw heatmap on
        fig2, ax = plt.subplots()




#next draw the heatmap and annotate with the predicted pnl
        sns.heatmap(df_put_pnl, annot=False, center = 0, cmap = LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256),annot_kws={'va':'bottom', "color" : "black"},fmt="", cbar=True )

        plt.title("Heatmap of predicted Pnl of a put option (£)", fontsize = 15) 
        plt.xlabel('Volatility', fontsize = 10) 
        plt.ylabel('Spot Price', fontsize = 10) 


######
#create columns to display the heatmaps in repsective and display
        col_call_map, col_put_map = st.columns(2)

        with col_call_map:
            st.pyplot(fig1)

        with col_put_map:
            st.pyplot(fig2)
        
    
    st.write("Note; we are changing the spot price and volatility as the Black Scholes equation is most sensitive to these variables.")