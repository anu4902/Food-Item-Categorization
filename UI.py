import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os
from nltk import word_tokenize
import pickle
import sklearn
import json
from openai import OpenAI

def nav1():
    df=pd.read_csv("updated_data.csv")
    df.drop(['Unnamed: 0','NewDesc'],axis=1,inplace=True)
    st.dataframe(df,height=600,width=800)
    
def process_json(filename):
    row={}
    path='FoodData/'+filename
    with open(path,'r') as f:
        dish_det=json.load(f)
        
        row['DishName']=os.path.splitext(filename)[0]
        row['Ingredients']=[i.strip() for i in dish_det['InputCols']['List of Main Ingredients'].split(",")]
        row['Calories']=dish_det['InputCols']['Nutritional Value per Serving']['Calories']
        row['SFat']=dish_det['InputCols']['Nutritional Value per Serving']['Saturated Fat in daily %']
        row['Sugar']=dish_det['InputCols']['Nutritional Value per Serving']['Sugar in g']
        row['Protein']=dish_det['InputCols']['Nutritional Value per Serving']['Protein in g']
        row['GI']=dish_det['InputCols']['Nutritional Value per Serving']['Glycemic Index']
        row['Region']=dish_det['InputCols']['Region where Dish is Popular']
        row['Description']=dish_det['InputCols']['Description']
        row['SpiceLevel']=dish_det['InputCols']['Spice Level']
        row['CookingMethod']=dish_det['InputCols']['Cooking Method']
        row['PrepTime']=dish_det['InputCols']['Preparation Time']
        row['HasAllergens']=dish_det['InputCols']['Contains Allergens']
        row['Cuisine']=dish_det['OutputCols']['Cuisine Type']
        row['MealType']=[i.strip() for i in dish_det['OutputCols']['Meal Type'].split(",")]
        row['DietaryPreference']=dish_det['OutputCols']['Dietary Preference']
        row['Protein-Rich']=dish_det['OutputCols']['Protein Rich']
        row['Pregnancy-Safe']=dish_det['OutputCols']['Pregnancy Safe']
        row['Diabetic-Friendly']=dish_det['OutputCols']['Diabetic Friendly']

    return row

def showimages(tags):
    st.write(tags)
    if "Veg" in tags or "Vegetarian" in tags:
        #st.write("Veg")
        st.image("TagImages/veg.jpg",width=100)
    if "Non-Veg" in tags or "Non-Vegetarian" in tags:
        #st.write("Non-Veg")
        st.image("TagImages/nonveg.png",width=100)
    if "Indian" in tags or "South Indian" in tags or "North Indian" in tags:
        #st.write("Indian food")
        st.image("TagImages/indian.jpeg",width=100)
    if "Continental" in tags:
        #st.write("Continental")
        st.image("TagImages/continental.jpeg",width=100)
    if "Protein-Rich" in tags:
        #st.write("Protein-Rich")
        st.image("TagImages/protein.jpg",width=100)
    if "Pregnancy-Safe" in tags:
        #st.write("Pregnancy-Safe")
        st.image("TagImages/pregnancy.jpeg",width=100)
    if "Diabetic-Friendly" in tags:
        #st.write("Diabetic-Friendly")
        st.image("TagImages/diabetic.jpg",width=100)
    if "Spicy" in tags:
        #st.write("Spicy")
        st.image("TagImages/spicy.png",width=100)
    if "Millet-Based" in tags:
        #st.write("Millet-Based")
        st.image("TagImages/millet.jpg",width=100)
    if "Seafood" in tags:
        #st.write("Seafood")
        st.image("TagImages/seafood.jpeg",width=100)
    if "May Contain Allergens" in tags:
        #st.write("May Contain Allergens")
        st.image("TagImages/allergen.jpeg",width=100)

    

def run_GPT4(document,model='gpt-3.5-turbo-0125'):
    client = OpenAI(api_key="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")

    try:
        prmpt = """
## Dish Information Generation Prompt

Generate a comprehensive dataset for a dish, including detailed input features and output classifications. Input features should encompass the main ingredients, nutritional content, region of popularity, a brief description, spice level, cooking method, preparation time, and allergen presence. Based on these inputs, output classifications will include the dish's cuisine type, meal type, dietary preference, and health considerations.

### Input Features to Generate
- **List of Main Ingredients**: Specify main components in string format.
- **Nutritional Value per Serving**: Provide values for calories, saturated fat (% daily value), sugar (g), protein (g), and glycemic index.
- **Region**: Name the region where the dish is popular.
- **Description**: Give a 100-word description.
- **Spice Level**: Rate from 0 to 10.
- **Cooking Method**: Describe the primary cooking method.
- **Preparation Time**: List in minutes.
- **Contains Allergens**: Indicate with true/false.

### Output Classifications
- **Cuisine Type**: The culinary tradition of the dish.
- **Meal Type**: Suitable times for consumption, allowing multiple values.
- **Dietary Preference**: Categorize as Veg, Non-Veg, or Vegan.
- **Protein Rich**: Indicate with true/false if high in protein.
- **Pregnancy Safe**: Mark with true/false based on safety for pregnancy.
- **Diabetic Friendly**: Show with true/false if suitable for diabetics.

### Expected JSON Format
```json
{
  "InputCols": {
    "List of Main Ingredients": "String",
    "Nutritional Value per Serving": {
      "Calories": "Integer",
      "Saturated Fat in daily %": "Integer",
      "Sugar in g": "Integer",
      "Protein in g": "Integer",
      "Glycemic Index": "Integer"
    },
    "Region where Dish is Popular": "String",
    "Description": "String",
    "Spice Level": "Integer",
    "Cooking Method": "String",
    "Preparation Time": "Integer",
    "Contains Allergens": "Boolean"
  },
  "OutputCols": {
    "Cuisine Type": "String",
    "Meal Type": "String",
    "Dietary Preference": "String",
    "Protein Rich": "Boolean",
    "Pregnancy Safe": "Boolean",
    "Diabetic Friendly": "Boolean"
  }
}

Ensure strict adherence to the specified JSON format for output. The `"InputCols"` section must accurately reflect the detailed input features of the dish, including ingredients, nutritional information, and other specified attributes. The `"OutputCols"` should precisely classify the dish's cuisine type, meal type, dietary preference, and health considerations based on the input provided. Each field in the output JSON must match the expected data types and structure, with boolean values for health considerations, integer values for nutritional content and spice level, and string values for all other categories. Accuracy in this structure is crucial for the utility and integrity of the generated dataset.

"""

        rspnse = client.chat.completions.create(model=model, response_format={"type": "json_object"},messages=[{'role':'system','content':prmpt},{'role': "user", 'content':"Based on the instructions, Provide output for Dish:\n"+document}])
        gpt_ans = rspnse.choices[0].message.content
        return gpt_ans

    except:
        print('TimeoutError')
        return 'openai-timeout'
    
def predict_model(new_df):
    cols_to_drop=['Description_Embeddings','CookingMethod_Embeddings','DishName','Ingredients','Region','MealType','CookingMethod','Description']
    num_cols=['Calories','SFat','Sugar','Protein','GI','SpiceLevel','PrepTime']
    word2vec_cols=['DishName','Ingredients','Region']
    sent_trans_cols=['Description','CookingMethod']
    dietary_cols=['DietaryPreference_Veg','DietaryPreference_Non-Veg']
    cuisine_cols=['NewCuisine_Indian','NewCuisine_Continental'] 

    with open('PickleModels/Scaler.pkl','rb') as f:
        scaler=pickle.load(f)

    sent_transform_models={}
    for i in sent_trans_cols:
        with open('PickleModels/'+i+'_sent_trans.pkl','rb') as f:
            sent_transform_models[i]=pickle.load(f)
    
    pca_models={}
    for i in sent_trans_cols:
        with open('PickleModels/'+i+'_pca.pkl','rb') as f:
            pca_models[i]=pickle.load(f)

    n_components={'Description':128,'CookingMethod':32}

    word2vec_models={}
    for i in word2vec_cols:
        with open('PickleModels/'+i+'_w2v.pkl','rb') as f:
            word2vec_models[i]=pickle.load(f)

    with open('PickleModels/DietaryPreference.pkl','rb') as f:
        dietary_model=pickle.load(f)

    with open('PickleModels/Cuisine.pkl','rb') as f:
        cuisine_model=pickle.load(f)

    with open('PickleModels/Protein-Rich.pkl','rb') as f:
        protein_model=pickle.load(f)

    with open('PickleModels/Diabetic-Friendly.pkl','rb') as f:
        diabetic_model=pickle.load(f)

    with open('PickleModels/Pregnancy-Safe.pkl','rb') as f:
        pregnancy_model=pickle.load(f)


    #numeric columns
    new_df[num_cols]=scaler.transform(new_df[num_cols])

    #text
    for col in sent_trans_cols:
        model=sent_transform_models[col]
        new_df[col+'_Embeddings'] = new_df[col].apply(lambda x: model.encode(x))

        new_embeddings=list(new_df[col+'_Embeddings'])
        
        pca=pca_models[col]
        new_pca_result=pca.transform(new_embeddings)
        
        # Create a DataFrame with PCA results
        cols=[col+'_PC'+str(i) for i in range(1,n_components[col]+1)]
        new_df_pca = pd.DataFrame(data=new_pca_result, columns=cols,index=new_df.index)
        new_df=pd.concat([new_df,new_df_pca],axis=1)

    for col in ['DishName','Region']:
        new_df[col] = new_df[col].apply(lambda x: word_tokenize(x.lower()))

    for col in word2vec_cols:

        word2vec_model=word2vec_models[col]
        # Function to generate text embeddings
        def text_to_embedding(tokens, model):
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            return sum(vectors) if vectors else [0] * model.vector_size  # Return zeros if no valid tokens

        new_df[col+'_Embedding'] = new_df[col].apply(lambda x: text_to_embedding(x, word2vec_model))
        
        # Flatten the text embeddings into multiple columns
        embedding_cols = [f'{col}_embed_{i+1}' for i in range(word2vec_model.vector_size)]
        new_df[embedding_cols] = pd.DataFrame(new_df[col+'_Embedding'].tolist(), index=new_df.index)
        
        # Drop the original 'text_embedding' column
        new_df = new_df.drop(columns=[col+'_Embedding'],axis=1)
        

    for i in cols_to_drop:
        if i in new_df.columns:
            new_df.drop(i,inplace=True,axis=1)

    for i in new_df.columns:
        if new_df[i].dtype=='bool':
            new_df[i]=new_df[i].astype('int')
            new_df[i]=new_df[i].astype('int')    

    x_cols=new_df.columns
    x_test_new=new_df[x_cols]
    ypred_protein=protein_model.predict(x_test_new)
    ypred_diabetic=diabetic_model.predict(x_test_new)
    ypred_preg=pregnancy_model.predict(x_test_new)
    ypred_diet=dietary_model.predict(x_test_new)
    ypred_cuisine=cuisine_model.predict(x_test_new)

    return ypred_protein,ypred_diet,ypred_preg,ypred_diabetic,ypred_cuisine


selected=st.sidebar.radio("Main Menu",['About','View Data','Visualize','Generate food description tags from DB','Get model predictions from info','Get description without providing info'])    

if selected =='About':
    st.title("Food Item Categorization")
    st.write('This project is dedicated to developing an intelligent system capable of automatically generating descriptive tags for various food items.')
    st.write('\n')
    st.write('Leveraging web mining, text mining, and various Machine Learning techniques, this project aims to streamline the process of creating engaging and contextually relevant food description tags, catering to the increasing demand for efficient content generation in the food industry.')
    st.write("\n\n")
    st.image("AboutImg.jpeg")
    st.write('\n')
    st.title("Workflow")
    st.image('DM_workflow.jpeg')

elif selected=='View Data':
    st.title("Food Items Description Data")
    nav1()
         
elif selected=='Visualize':
    plots={}
    directory_path = '/Users/anu/Documents/NLP_Project_Food/Plots/'
    for filename in os.listdir(directory_path):
        name , file_extension = os.path.splitext(filename)

        if name=='wc_cookingmethod':
            plots['Cooking Methods']=directory_path+filename
        elif name=='wc_desc':
            plots['Description']=directory_path+filename
        elif name=='wc_ing':
            plots['Ingredients']=directory_path+filename
        elif name=='wc_regions':
            plots['Regions']=directory_path+filename
        elif name=='cp_allergen':
            plots['Allergens']=directory_path+filename
        else:
            plots['Spice Level'] = directory_path+filename
    
    st.title("Visualization")
    st.write("Word clouds and data distributions of various features:\n")

    for i in ['Ingredients','Cooking Methods','Description','Regions','Allergens','Spice Level']:
        st.title(i)
        st.image(plots[i])
        st.write('\n\n')
        
elif selected=='Generate food description tags from DB':
    st.title("Generate tags for your own dish!")

    df=pd.read_csv("updated_data.csv")
    df.drop(['Unnamed: 0','NewDesc'],axis=1,inplace=True)
    dishname=st.text_input("Enter dish name: ")
    
    if st.button("Get descriptive tags"):
        tags=[]
        if dishname in df['DishName'].unique():
            new_df=df[df['DishName']==dishname]
            #st.write(new_df)
            new_data_ing=new_df['Ingredients'].unique()[0]
        
            #print(new_df)
            
            seafood=['Fish','Shrimp','Prawns','Crab','Lobster','Clams','Mussels','Oysters','Scallops','Squid']
            for i in new_data_ing:
                if i in seafood:
                    tags.append('Seafood')
                    break

            if 'Chicken' in new_data_ing[:2]:
                tags.append('Chicken')

            if 'Egg' in new_data_ing[:2] or 'Eggs' in new_data_ing[:2]:
                tags.append("Eggs")

            if 'Mutton' in new_data_ing[:2]:
                tags.append('Mutton')

            millets=['jowar', 'bajra', 'ragi flour', 'ragi', 'kangni', 'kutki', 'jhangora', 'kodra', 'cheena', 'korle']
            #st.write(new_data_ing[0])
            for i in new_data_ing[:5]:
                if i.lower() in millets:
                    tags.append('Millet-Based')

            if (new_df['SpiceLevel']>5).any():
                tags.append('Spicy')

            if (new_df['HasAllergens']==1).any():
                tags.append("May Contain Allergens")

            if (new_df['Protein-Rich']==1).any():
                tags.append("Protein-Rich")

            if (new_df['Pregnancy-Safe']==1).any():
                tags.append("Pregnancy-Safe")

            if (new_df['Diabetic-Friendly']==1).any():
                tags.append("Diabetic-Friendly")
            
            #st.write(new_df['Cuisine'])
            tags.append(new_df['DietaryPreference'].iloc[0])
            tags.append(new_df['Cuisine'].iloc[0])
            
            #for t in tags:
            #    st.write(t)
            showimages(tags)
            
            
        else:   
            st.write("Dish doesn't exist in the database. Use the model predictions tab!")

elif selected=='Get model predictions from info':
        st.title("Generate tags for your own dish by using model predictions!")
   
        new_df={}
        new_df['DishName']=st.text_input("Dish Name")
        new_data_ing=st.text_input("Main Ingredients as comma separated values: ").split(",")
        new_data_ing=[i.strip() for i in new_data_ing]
        new_df['Ingredients']=new_data_ing
        st.write("\nNutritional info per serving:\n")
        new_df['Calories']=st.number_input("Calories")
        new_df['SFat']=st.number_input("Saturated fat in g")
        new_df['Sugar']=st.number_input("Sugar in g")
        new_df['Protein']=st.number_input("Protein in g")
        new_df['GI']=st.number_input("Glycemic index")
        new_df['SpiceLevel']=st.number_input("Spice Level")
        new_df['Region']=st.text_input("Region where the dish is popular")
        new_df['Description']=st.text_area("A short description about the dish")
        new_df['CookingMethod']=st.text_input("Cooking Method")
        new_df['PrepTime']=st.number_input("Approximate time taken to prepare the dish in mins: ")
        new_df['HasAllergens']=1 if st.radio("Does the dish contain any allergens?",['Yes','No'])=='Yes' else 0
                
        new_df=pd.DataFrame(new_df)
        ypred_protein,ypred_diet,ypred_preg,ypred_diabetic,ypred_cuisine=predict_model(new_df)

        tags=[]
        seafood=['fish','shrimp','prawns','crab','lobster','clams','mussels','oysters','scallops','squid']
        for i in new_data_ing:
            if i.strip().lower() in seafood:
                tags.append('Seafood')
                break

        if 'Chicken' in new_data_ing[:2] or 'chicken' in new_data_ing[:2]:
            tags.append('Chicken')

        if 'Egg' in new_data_ing[:2] or 'Eggs' in new_data_ing[:2] or 'egg' in new_data_ing[:2]:
            tags.append("Eggs")

        if 'Mutton' in new_data_ing[:2] or 'mutton' in new_data_ing[:2]:
            tags.append('Mutton')

        millets=['jowar', 'bajra', 'ragi', 'kangni', 'kutki', 'jhangora', 'kodra', 'cheena', 'korle']
        for i in new_data_ing[:5]:
            if i.lower() in millets:
                tags.append('Millet-Based')

        if (new_df['SpiceLevel']>1).any():
            tags.append('Spicy')


        if (new_df['HasAllergens']==1).any():
            tags.append("May Contain Allergens")
        
        if ypred_protein[0]:
            tags.append("Protein-Rich")

        if ypred_preg[0]:
            tags.append("Pregnancy-Safe")

        if ypred_diabetic[0]:
            tags.append("Diabetic-Friendly")

        dietary_cols=['DietaryPreference_Non-Veg','DietaryPreference_Veg']
        diet=None
        for i in range(len(ypred_diet[0])):
            if ypred_diet[0][i]==1:
                diet=dietary_cols[i].split('_')[1]
        if diet is not None:
            tags.append(diet)
                
        cuisine_cols=['NewCuisine_Continental','NewCuisine_Indian'] 
        cuisine=None
        for i in range(len(ypred_cuisine[0])):
            if ypred_cuisine[0][i]==1:
                cuisine=cuisine_cols[i].split('_')[1]
                break 
        if cuisine is not None:
            tags.append(cuisine)

        if st.button("Get tags"):
            #for t in tags:
            #    st.write(t)                  
            showimages(tags)

elif selected=='Get description without providing info':
        st.title("Generate tags for your own dish without providing any info!")

        dishname=st.text_input("Enter dish name: ")
        if st.button("Generate tags"):

            response=run_GPT4(dishname)
            while response=='openai-timeout':
                print("Trying again")
                response=run_GPT4(dishname)

            res_d=json.loads(response)
            json_obj=json.dumps(res_d,indent=2)

            filename=dishname+".json"
            with open('FoodData/'+filename,"w") as f:
                f.write(json_obj)
                    
            row=process_json(dishname+'.json')
            new_df=pd.DataFrame(columns=['DishName','Ingredients','Calories','SFat','Sugar','Protein','GI','Region','Description','SpiceLevel','CookingMethod','PrepTime','HasAllergens','Cuisine','MealType','DietaryPreference','Protein-Rich','Pregnancy-Safe','Diabetic-Friendly'])
            new_df.loc[len(new_df)]=row

            tags=[]
            new_data_ing=new_df['Ingredients'].tolist()[0]

            seafood=['fish','shrimp','prawn', 'prawns','crab','lobster','clams','mussels','oysters','scallops','squid']
            for i in new_df['Ingredients'].tolist()[0]:
                if i.lower() in seafood:
                    tags.append('Seafood')
                    break

            if 'Chicken' in new_data_ing[:2]:
                tags.append('Chicken')

            if 'Egg' in new_data_ing[:2] or 'Eggs' in new_data_ing[:2]:
                tags.append("Eggs")

            if 'Mutton' in new_data_ing[:2]:
                tags.append('Mutton')

            millets=['Jowar', 'Bajra', 'Ragi', 'Kangni', 'Kutki', 'Jhangora', 'Kodra', 'Cheena', 'Korle']
            for i in new_data_ing[:5]:
                if i in millets:
                    tags.append('Millet-Based')

            if (new_df['SpiceLevel']>5).any():
                tags.append('Spicy')

            if (new_df['HasAllergens']==1).any():
                tags.append("May Contain Allergens")

            if new_df['Protein-Rich'][0]:
                tags.append("Protein-Rich")

            if new_df['Pregnancy-Safe'][0]:
                tags.append("Pregnancy-Safe")

            if new_df['Diabetic-Friendly'][0]:
                tags.append("Diabetic-Friendly")

            tags.append(new_df['DietaryPreference'][0])
            tags.append(new_df['Cuisine'][0])
                    
            meal_type='/'.join(new_df['MealType'][0])
            tags.append(meal_type)
            
            #for t in tags:
            #    st.write(t)              
            showimages(tags)