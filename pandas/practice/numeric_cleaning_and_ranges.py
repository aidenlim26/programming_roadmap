import pandas as pd
df = pd.read_csv("Building_Permits.csv")
#Objective: Detect and correct bad numeric values.
#Check for negative or zero values in “Price” or “Volume”.
#Replace invalid values (e.g., negative prices) with np.nan.
#Detect anomalies using simple rules (e.g., prices > 10× median price).
#Clip extreme values using clip(lower, upper).