# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:45:36 2018

@author: victzuan
"""

import media_OOP
import fresh_tomatos

toy_story = media_OOP.Movie("Toy Story", 
                            "A story about a boy and his toys that come to life",
                            "http:/upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg",
                            "https://www.youtube.com/watch?v=KYz2wyBy3kc"
                            )

#print(toy_story.storyline)

avatar = media_OOP.Movie("Avatar",
                         "A marine on a alien planet",
                         "https://en.wikipedia.org/wiki/Avatar_(2009_film)#/media/File:Avatar-Teaser-Poster.jpg",
                         "https://www.youtube.com/watch?v=5PSNL1qE6VY"
                         )

#print(avatar.storyline)
#avatar.show_trailer()

o_mecanismo = media_OOP.Movie("O Mecanismo",
                              "How the biggest corruption scandal got unfolded by the Brazilian Federal Police",
                              "https://en.wikipedia.org/wiki/The_Mechanism_(TV_series)#/media/File:O_Mecanismo_official_artwork.jpg",
                              "https://www.youtube.com/watch?v=13OtvUxOcUU"
                              )
#o_mecanismo.show_trailer()

movies = [toy_story, avatar, o_mecanismo]

fresh_tomatos.open_movies_page(movies)
#print (media_OOP.Movie.VALID_RATINGS)
#print(media_OOP.Movie.__doc__)