# AI Resume Sorting
Given a custom resume, predicts which category it belongs to. From the following categories <br><br>```['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing',
                                                                                  'Mechanical Engineer', 'Sales', 'Health and fitness',
                                                                                  'Civil Engineer', 'Java Developer', 'Business Analyst',
                                                                                  'SAP Developer', 'Automation Testing', 'Electrical Engineering',
                                                                                  'Operations Manager', 'Python Developer', 'DevOps Engineer',
                                                                                  'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
                                                                                  'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']
                                                                                 ```

## Usage
1. Download and open the colab notebook provided.
2. Download and extract the csv file containing data locally from the dataset
3. Upload this csv file to your colab notebook <br> ![image](https://github.com/user-attachments/assets/b8479d70-b7b8-4f04-aeb6-4bdf47f7be6c)
4. Run all cells
5. Call ```get_resume_class(YOUR_RESUME)``` in a new cell and replace YOUR_RESUME with your resume text (Add it in triple quotes as resumes can be quite long)

## Dataset
https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
