Web Scraping with Selenium and BeautifulSoup
=============================================

This project automates the process of logging into a website, reading a list of URLs from a CSV file, and scraping content from each URL using Selenium and BeautifulSoup.

Requirements
------------
Before running the script, make sure you have the following installed:

- Python 3.x
- ChromeDriver (compatible with your version of Chrome)*
- Required Python libraries (are installed within the code)

*You need to download ChromeDriver for Selenium to work with your Chrome browser. You can find it at: https://sites.google.com/chromium.org/driver/

Setup
-----
1. Clone this repository or download the project files.
2. Place the chromedriver executable in the same directory as your script or specify its path directly in the code. (Please add the path to a .env file located in the same directory as the project with the name CHROMEDRIVER)
3. Ensure the InterestingLinks.csv (with the columns: url and type) file is located in the same directory as the script. This CSV file should contain URLs and other relevant data for scraping. 
4. Modify the email and password variables with your own credentials for logging into the website. (Please add them to a .env file located in the same directory as the project, with the name EMAIL and PASSWORD.)

How to Use
-----------
Run the code "WebScrapping.ipynb"


Code Walkthrough
----------------
1. Start Session:
   login_url = 'https://cursos.desafiolatam.com/users/sign_in'
   driver = start_session(login_url)
   time.sleep(5)
   - The start_session function is called to open the browser and log in to the specified URL.
   - The script waits for 5 seconds to ensure the page has fully loaded after logging in.

2. Read CSV File:
   files = pd.read_csv("InterestingLinks.csv")
   - The InterestingLinks.csv file is read into a DataFrame. This file should contain URLs and other metadata for scraping.

3. Scraping:
   scraping(files, driver)
   - The scraping function is called to scrape content from each URL in the files DataFrame using the driver (the Selenium WebDriver instance).

4. Scraping Logic:
   - The script checks the type of URL (e.g., free_course, paid_course, or free_tutorial), navigates to the page, and extracts specific content based on predefined sections or classes.
   - The extracted content is saved as a .txt file.

File Structure
--------------
project-directory/
│
├── chromedriver/                # ChromeDriver executable
    ├──chromedriver.exe
├── courses_web_page             # Directory with scrapping results
├── InterestingLinks.csv         # CSV file containing URLs for scraping
├── WebScrapping.ipynb           # Main code        
└── requierements.txt            # Library with the neccesary libraries

Notes
-----
- Make sure your InterestingLinks.csv file contains the correct URLs to scrape.
- The scraping logic depends on the structure of the web pages you are scraping, so if the website's structure changes, you may need to adjust the scraping logic accordingly.

License
-------
This project is licensed under the MIT License - see the LICENSE file for details.


