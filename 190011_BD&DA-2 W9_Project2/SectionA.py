
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test.head()

train.columns
test.columns

train.dtypes

print("Train dataset shape ", train.shape)
train.head()

print('Test dataset shape', test.shape)
test.head()

train["Loan_Status"].count()
train["Loan_Status"].value_counts()
train['Loan_Status'].value_counts(normalize=True)*100

train['Loan_Status'].value_counts(normalize=True).plot.bar(title='Loan Status')
plt.ylabel('Loan Applicants ')
plt.xlabel('Status')
plt.show()

print("The loan of 422(around 69%) people out of 614 was approved.")

print("Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)")

train["Gender"].count()
train['Gender'].value_counts()
train['Gender'].value_counts(normalize=True)*100
train['Gender'].value_counts(normalize=True).plot.bar(title ='Gender of loan applicant data')

print("Ques 1(a): Find out the number of male and female in loan applicants data.")
plt.xlabel('Gender')
plt.ylabel('Number of loan applicant')
plt.show()

print("Ans 1(a): ")
print( "Gender variable contain Male : 81% Female: 19% Male")

train["Married"].count()
train["Married"].value_counts()

print("Total number of people: 611")
print("Married: 398")
print("Unmarried: 213")

train['Married'].value_counts(normalize=True)*100

print("Ques1(b): Find out the number of married and unmarried loan applicants.")
train['Married'].value_counts(normalize=True).plot.bar(title='Married Status of an applicant')
plt.xlabel('Married Status')
plt.ylabel('Number of loan applicant')
plt.show()

print("Ans 1(b)")
print("Number of married people : 65%")
print("Number of unmarried people : 35%")

train["Self_Employed"].count()
train['Self_Employed'].value_counts()
train['Self_Employed'].value_counts(normalize=True)*100

print("Ques1(c): Find out the overall dependent status in the dataset.")
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Dependent Status')
plt.xlabel('Dependent Status')
plt.ylabel('Number of loan applicant')
plt.show()

print("Ans1(c): ")
print("Among 582 people only 14% are Self_Employed and rest of the 86% are Not_Self_Employed")

train["Education"].count()
train["Education"].value_counts()
train["Education"].value_counts(normalize=True)*100


print(" Ques1(d): Find the count how many loan applicants are graduate and non graduate.")
train["Education"].value_counts(normalize=True).plot.bar(title = "Education")
plt.xlabel('Dependent Status')
plt.ylabel('Percentage')
plt.show()


print("Ans1(d)")
print("Total number of People : 614")
print("78% are Graduated and 22% are not Graduated ")

train["Property_Area"].count()
train["Property_Area"].value_counts()
train["Property_Area"].value_counts(normalize=True)*100


print("Ques1(e): Find out the count how many loan applicants property lies in urban, rural and semi-urban areas.")
train["Property_Area"].value_counts(normalize=True).plot.bar(title="Property_Area")
plt.xlabel('Area Type')
plt.ylabel('Percentage')
plt.show()


# In[40]:


print("Ans1(e)")
print("38% people from Semiurban area")
print("33% people from Urban area")
print("29% people from Rural area")

train["Credit_History"].count()
train["Credit_History"].value_counts()
train['Credit_History'].value_counts(normalize=True)*100

train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_History')
plt.xlabel('Debt')
plt.ylabel('Percentage')
plt.show()

print("Ques3")
print("To visualize and plot the distribution plot of all numerical attributes of the given train dataset i.e. ApplicantIncome,  CoApplicantIncome and LoanAmount.     ")

print("ApplicantIncome distribution")
plt.figure(1)
plt.subplot(121)
sns.distplot(train["ApplicantIncome"]);

plt.subplot(122)
train["ApplicantIncome"].plot.box(figsize=(16,5))
plt.show()


train.boxplot(column='ApplicantIncome',by="Education" )
plt.suptitle(" ")
plt.show()


print("Co-applicant Income distribution")
plt.figure(1)
plt.subplot(121)
sns.distplot(train["Co-applicantIncome"]);

plt.subplot(122)
train["Co-applicantIncome"].plot.box(figsize=(16,5))
plt.show()

print("Loan Amount Variable")
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))

plt.show()

print("Loan Amount Term Distribution")
plt.figure(1)
plt.subplot(121)
df = train.dropna()
sns.distplot(df["Loan_Amount_Term"]);

plt.subplot(122)
df["Loan_Amount_Term"].plot.box(figsize=(16,5))
plt.show()

print("Ques4")
print("Relation between Loan_Status and Gender")
print(pd.crosstab(train["Gender"],train["Loan_Status"]))
Gender = pd.crosstab(train["Gender"],train["Loan_Status"])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.show()

print("Conclusion of Relation between Loan_Status and Gender")
print("Number of Female whose Loan was approved : 75")
print("Number of Male whose Loan was approved :339 ")
print("Number of Female whose Loan was not0 approved : 37")
print("Number of Female whose Loan was approved : 150")
print("Proportion of Male applicants is higher for the approved loans.")


print("Relation between Loan_Status and Married status")
print(pd.crosstab(train["Married"],train["Loan_Status"]))
Married=pd.crosstab(train["Married"],train["Loan_Status"])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Married")
plt.ylabel("Percentage")
plt.show()

print("Conclusion of relation between Loan_Status and Married status")
print("Number of married people whose Loan was approved : 285")
print("Number of married people whose Loan was not approved : 113")
print("Number of unmarried people whose Loan was approved : 134")
print("Number of unmarried people whose Loan was not approved : 79")
print("Proportion of Married applicants is higher for the approved loans.")

print("Relation between Loan_Status and Dependents")
print(pd.crosstab(train['Dependents'],train["Loan_Status"]))
Dependents = pd.crosstab(train['Dependents'],train["Loan_Status"])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Dependents")
plt.ylabel("Percentage")
plt.show()


print("Conclusion of relation between Loan_Status and Dependents")
print("Number of dependents on the loan applicant : 0 and Loan was approved : 238")
print("Number of dependents on the loan applicant : 0 and Loan was not approved : 107")
print("Number of dependents on the loan applicant : 1 and Loan was approved : 66")
print("Number of dependents on the loan applicant : 1 and Loan was not approved : 36")
print("Number of dependents on the loan applicant : 2 and Loan was approved : 76")
print("Number of dependents on the loan applicant : 2 and Loan was not approved : 25")
print("Number of dependents on the loan applicant : 3+ and Loan was approved : 33")
print("Number of dependents on the loan applicant : 3+ and Loan was not approved : 18")
print("Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status.")

print("Relation between Loan_Status and Education")
print(pd.crosstab(train["Education"],train["Loan_Status"]))
Education = pd.crosstab(train["Education"],train["Loan_Status"])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Education")
plt.ylabel("Percentage")
plt.show()

print("Conclusion of relation between Loan_Status and Education.")
print("Number of people who are Graduate and Loan was approved : 340")
print("Number of people who are Graduate and Loan was no approved : 140")
print("Number of people who are Not Graduate and Loan was approved : 82")
print("Number of people who are Not Graduate and Loan was not approved : 52")
print("Proportion of Graduate applicants is higher for the approved loans.")

print("Relation between Loan_Status and Self_Employed")
print(pd.crosstab(train["Self_Employed"],train["Loan_Status"]))
SelfEmployed = pd.crosstab(train["Self_Employed"],train["Loan_Status"])
SelfEmployed.div(SelfEmployed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Self_Employed")
plt.ylabel("Percentage")
plt.show()


print("Relation between Loan_Status and Self_Employed")
print("People who are Self_Employed and Loan was approed : 56")
print("People who are Self_Employed and Loan was not approed : 26")
print("People who are not Self_Employed and Loan was approed : 343")
print("People who are not Self_Employed and Loan was not approed : 157")
print("There is nothing significant we can infer from Self_Employed vs Loan_Status plot.")


print("Relation between Loan_Status and Credit_History")
print(pd.crosstab(train["Credit_History"],train["Loan_Status"]))
CreditHistory = pd.crosstab(train["Credit_History"],train["Loan_Status"])
CreditHistory.div(CreditHistory.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Credit_History")
plt.ylabel("Percentage")
plt.show()

print("Conclusion relation between Loan_Status and Credit_History")
print("People with credit history as 1 and loan was approved : 378")
print("People with credit history as 1 and loan was not approved : 97")
print("People with credit history as 0 and loan was approved : 7")
print("People with credit history as 0 and loan was not approved : 82")
print("It seems people with credit history as 1 are more likely to get their loans approved.")


print("Relation between Loan_Status and Property_Area")
print(pd.crosstab(train["Property_Area"],train["Loan_Status"]))
PropertyArea = pd.crosstab(train["Property_Area"],train["Loan_Status"])
PropertyArea.div(PropertyArea.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Property_Area")
plt.ylabel("Loan_Status")
plt.show()


print("Relation between Loan_Status and Property_Area")
print("People who are from Rural area and loan was approved : 110")
print("People who are from Rural area and loan was not approved : 69")
print("People who are from Semiurban area and loan was approved : 179")
print("People who are from Semiurban area and loan was not approved : 54")
print("People who are from Urban area and loan was approved : 133")
print("People who are from Semiurban area and loan was not approved : 69")
print("Proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas")

print("Relation between Loan_Status and Income")
train.groupby("Loan_Status")['ApplicantIncome'].mean().plot.bar(title= " Relation between ApplicaantIncome and Loan Status")
plt.ylabel(" Applicant Income")


bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)


print(pd.crosstab(train["Income_bin"],train["Loan_Status"]))
Income_bin = pd.crosstab(train["Income_bin"],train["Loan_Status"])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("ApplicantIncome")
plt.ylabel("Percentage")
plt.show()

print("Applicant income does not affect the chances of loan approval ")


bins=[0,1000,3000,42000]
group =['Low','Average','High']
train['CoapplicantIncome_bin']=pd.cut(df["CoapplicantIncome"],bins,labels=group)


print(pd.crosstab(train["CoapplicantIncome_bin"],train["Loan_Status"]))
CoapplicantIncome_Bin = pd.crosstab(train["CoapplicantIncome_bin"],train["Loan_Status"])
CoapplicantIncome_Bin.div(CoapplicantIncome_Bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.xlabel("CoapplicantIncome")
plt.ylabel("Percentage")
plt.show()

print("It shows that if co-applicant’s income is less the chances of loan approval are high. But this does not look right. The possible reason behind this may be that most of the applicants don’t have any co-applicant so the co-applicant income for such applicants is 0 and hence the loan approval is not dependent on it. So we can make a new variable in which we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval.")

train["TotalIncome"]=train["ApplicantIncome"]+train["CoapplicantIncome"]

bins =[0,2500,4000,6000,81000]
group=['Low','Average','High','Very High']
train["TotalIncome_bin"]=pd.cut(train["TotalIncome"],bins,labels=group)


print(pd.crosstab(train["TotalIncome_bin"],train["Loan_Status"]))
TotalIncome = pd.crosstab(train["TotalIncome_bin"],train["Loan_Status"])
TotalIncome.div(TotalIncome.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(2,2))
plt.xlabel("TotalIncome")
plt.ylabel("Percentage")
plt.show()

print("Proportion of loans getting approved for applicants having low Total_Income is very less as compared to that of applicants with Average, High and Very High Income.")

print("Whose TotalIncome was Low and loan was approved : 10")
print("Whose TotalIncome was Low and loan was not approved : 14")
print("Whose TotalIncome was Average and loan was approved : 87")
print("Whose TotalIncome was Average and loan was not approved : 32")
print("Whose TotalIncome was High and loan was approved : 159")
print("Whose TotalIncome was High and loan was not approved : 65")
print("Whose TotalIncome was Very High and loan was approved : 166")
print("Whose TotalIncome was Very High and loan was not approved : 81")

print("Relation between Loan_Status and Loan Amount")

bins = [0,100,200,700]
group=['Low','Average','High']
train["LoanAmount_bin"]=pd.cut(df["LoanAmount"],bins,labels=group)

print(pd.crosstab(train["LoanAmount_bin"],train["Loan_Status"]))
LoanAmount=pd.crosstab(train["LoanAmount_bin"],train["Loan_Status"])
LoanAmount.div(LoanAmount.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.xlabel("LoanAmount")
plt.ylabel("Percentage")
plt.show()


print("Whose Loan Amount was low and Loan was approved : 86")
print("Whose Loan Amount was low and Loan was not approved : 38")
print("Whose Loan Amount was Average and Loan was approved : 207")
print("Whose Loan Amount was Average and Loan was not approved : 83")
print("Whose Loan Amount was High and Loan was approved : 39")
print("Whose Loan Amount was High and Loan was not approved : 27")
print("Conclusion:")
print(" The proportion of approved loans is higher for Low and Average Loan Amount as compared to that of High Loan Amount")


train=train.drop(["Income_bin","CoapplicantIncome_bin","LoanAmount_bin","TotalIncome","TotalIncome_bin"],axis=1)


train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)


matrix = train.corr()
f, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu",annot=True);

print("We will use the heat map to visualize the co-relation. Heatmaps visualize data through variations in coloring. The variables with darker color means their correlation is more.")
print("We see that the most correlated variables are (ApplicantIncome - LoanAmount) and (Credit_History - Loan_Status).")
