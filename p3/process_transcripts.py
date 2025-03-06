# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2024
# Project Part 3
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment then you need prior approval from course staff.
# =========================================================================================================

import string
import re
import csv
import nltk


# Function: process_transcripts(fname)
# fname: A string indicating a file name
# Returns: Nothing (writes output to file)
#
# This function processes a provided transcript file by creating three versions of it:
# one includes all utterances, with one utterance per line; another includes only the
# chatbot utterances; and the third includes only the user utterances.  None of these
# files should contain the speaker tags ("CHATBOT" or "USER").
def process_transcripts(fname):
    f_in = open(fname, "r")
    f_out_all = open("all_{0}".format(fname), "w")
    f_out_chatbot = open("chatbot_{0}".format(fname), "w")
    f_out_user = open("user_{0}".format(fname), "w")

    # [WRITE YOUR CODE HERE]

    
    f_in.close()
    f_out_all.close()
    f_out_chatbot.close()
    f_out_user.close()

    return



# This is your main() function.  Use this space to try out and debug your code
# using your terminal.  The code you include in this space will not be graded.  If
# you run this code using test.txt as the input file, the contents of the output
# files it produces will be as follows.
#
# all_test.txt:
# Welcome to the CS 421 chatbot!  What is your name?
# Natalie Parde
# Thanks Natalie Parde!  What do you want to talk about today?
# I'm excited that it's a new semester!
# It sounds like you're in a positive mood!
# I'd also like to do a quick stylistic analysis. What's on your mind today?
# I'm currently creating a sample transcript for the CS 421 students.  This will help them ensure that their programs work correctly.  It will also provide an example interaction for them!
# Thanks!  Here's what I discovered about your writing style.
# Type-Token Ratio: 0.8823529411764706
# Average Tokens Per Sentence: 11.333333333333334
# # Nominal Subjects: 5
# # Direct Objects: 2
# # Indirect Objects: 0
# # Nominal Modifiers: 1
# # Adjectival Modifiers: 0
# Custom Feature #1: 0
# Custom Feature #2: 5
# What would you like to do next?  You can quit, redo the sentiment analysis, or redo the stylistic analysis.
# I think I'd like to quit.
#
# chatbot_test.txt:
# Welcome to the CS 421 chatbot!  What is your name?
# Thanks Natalie Parde!  What do you want to talk about today?
# It sounds like you're in a positive mood!
# I'd also like to do a quick stylistic analysis. What's on your mind today?
# Thanks!  Here's what I discovered about your writing style.
# Type-Token Ratio: 0.8823529411764706
# Average Tokens Per Sentence: 11.333333333333334
# # Nominal Subjects: 5
# # Direct Objects: 2
# # Indirect Objects: 0
# # Nominal Modifiers: 1
# # Adjectival Modifiers: 0
# Custom Feature #1: 0
# Custom Feature #2: 5
# What would you like to do next?  You can quit, redo the sentiment analysis, or redo the stylistic analysis.
#
# user_test.txt:
# Natalie Parde
# I'm excited that it's a new semester!
# I'm currently creating a sample transcript for the CS 421 students.  This will help them ensure that their programs work correctly.  It will also provide an example interaction for them!
# I think I'd like to quit.
if __name__ == "__main__":
    fname = "test.txt"
    process_transcripts(fname)