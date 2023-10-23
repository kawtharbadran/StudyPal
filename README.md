Study Pal to help with studying, because misery loves company. :')

Have Python already installed with this. Python 3.11.4 was used for this project.

SOURCE:
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/orchestration/langchain/intro_langchain_palm_api.ipynb 
https://github.com/themlguy-tf/generative-ai/blob/main/%233b_%5BNJ%5D_UseCases-PDF%2C%20Sequential%20%26%20Summarization_LangChain_%26_GCP_PaLM_.ipynb

VENV:
First, create an empty folder that will be your working folder. Open it with Visual Studio.
We want to create a virtual environment. Please check this link:
https://code.visualstudio.com/docs/python/environments 


AUTHENTICATION:
Please follow these instructions. You want to have an Application Default Credentials (ADC) in the end.
https://cloud.google.com/vertex-ai/docs/workbench/reference/authentication#client-libs:~:text=the%20command%20line-,Client%20libraries%20or%20third%2Dparty%20tools,-Set%20up%20Application

PACKAGES:
Make sure you install the required packages from requirements.txt.
After creating and activating venv, run this command:  pip install -r requirements.txt 

BEFORE RUNNING:
Change the vector_save_directory to where you want to save your PDF data.
Change the project ID to your own account on which you got authenticated.
