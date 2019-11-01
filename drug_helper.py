#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:02:23 2019

@author: flatironschool
"""

def category_converter(df):
    
    '''
    This function converts back the scaled categories.
    '''
    
    ages = {-0.95197: '18-24',
            -0.07854: '25-34',
            0.49788: '35-44',
            1.09449: '45-54',
            1.82213: '55-64',
            2.59171: '65+'}
    df.Age = df.Age.apply(lambda x: ages[round(x, 5)])
    
    
    genders = {0.48246: 'Female', -0.48246: 'Male'}

    df.Gender = df.Gender.apply(lambda x: genders[round(x, 5)])
    
    edus = {-2.43591: 'Left school before 16 years',
            -1.73790: 'Left school at 16 years',
            -1.43719: 'Left school at 17 years',
            -1.22751: 'Left school at 18 years',
            -0.61113: 'Some college or university, no certificate or degree',
            -0.05921: 'Professional certificate/ diploma',
            0.45468: 'University degree',
            1.16365: 'Masters degree',
            1.98437: 'Doctorate degree'}
    
    df.Education = df.Education.apply(lambda x: edus[round(x,5)])
    
    
    countries = {-0.09765: 'Australia',
                 0.24923: 'Canada',
                 -0.46841: 'New Zealand',
                 -0.28519: 'Other',
                 0.21128: 'Republic of Ireland',
                 0.96082: 'UK',
                 -0.57009: 'USA'}
    
    df.Country = df.Country.apply(lambda x: countries[round(x,5)])
    
    races = {-0.50212: 'Asian',
             -1.10702: 'Black',
             1.90725: 'Mixed-Black/Asian',
             0.12600: 'Mixed-White/Asian',
             -0.22166: 'Mixed-White/Black',
             0.11440: 'Other',
             -0.31685: 'White'}
    df.Ethnicity = df.Ethnicity.apply(lambda x: races[round(x,5)])
    
    
