# Contributing to d.abstract

New to contributing? Check [this](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

## How to contribute
1. Fork the project's repository. Click on the 'fork' button at the top of the page, which creates a copy of the repository on your own account. 
2. Clone that repository in a folder of your choice:
    
    git clone https://github.com/my-username/dabstract.git

3. Create a new branch where you add your contribution:
    
    git checkout -b name-of-contribution
    
4. When contributing make sure to follow the [code guidelines](#code-guidelines).

5. Stage your changes, commit and push them to the repository:
   
   git add file1.ext1 file2.ext2
   gut commit
   git push -u origin name-of-contribution

6. Go to your forked repository at https://github.com/my-username/dabstract and do a 'pull request'. 
   Your changes will be reviewed. 
   
## Code guidelines
1. Before creating a pull request first execute the existing [unit tests](tests/README.md). 
2. If your contribution is a new feature, please create a new unit test. 
3. We use [`black`](https://black.readthedocs.io/en/stable/) as code formatting style. Please use it as a pluging to your [code editor](https://black.readthedocs.io/en/stable/editor_integration.html).
   and run black on dabstract.
   
    black dabstract/
4. Documentation is crucial to proper usage and understanding of the code. Use numpy formatted docstring for each module. 
