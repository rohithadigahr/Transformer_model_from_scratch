"""
Various text collections for training the Transformer model
Choose different genres and styles for diverse training data
"""

class TextCollections:
    
    @staticmethod
    def get_classic_literature():
        """Classic literature excerpts (public domain)"""
        return """
        It was the best of times it was the worst of times it was the age of wisdom
        it was the age of foolishness it was the epoch of belief it was the epoch of
        incredulity it was the season of light it was the season of darkness it was
        the spring of hope it was the winter of despair we had everything before us
        we had nothing before us we were all going direct to heaven we were all going
        direct the other way in short the period was so far like the present period
        that some of its noisiest authorities insisted on its being received for good
        or for evil in the superlative degree of comparison only
        
        Call me Ishmael some years ago never mind how long precisely having little
        or no money in my purse and nothing particular to interest me on shore
        I thought I would sail about a little and see the watery part of the world
        it is a way I have of driving off the spleen and regulating the circulation
        whenever I find myself growing grim about the mouth whenever it is a damp
        drizzly November in my soul whenever I find myself involuntarily pausing
        before coffin warehouses and bringing up the rear of every funeral I meet
        
        In a hole in the ground there lived a hobbit not a nasty dirty wet hole
        filled with the ends of worms and an oozy smell nor yet a dry bare sandy
        hole with nothing in it to sit down on or to eat it was a hobbit hole
        and that means comfort it had a perfectly round door like a porthole
        painted green with a shiny yellow brass knob in the exact middle
        """ * 3
    
    @staticmethod
    def get_poetry_collection():
        """Poetry and lyrical content"""
        return """
        Two roads diverged in a yellow wood and sorry I could not travel both
        and be one traveler long I stood and looked down one as far as I could
        to where it bent in the undergrowth then took the other as just as fair
        and having perhaps the better claim because it was grassy and wanted wear
        though as for that the passing there had worn them really about the same
        
        Shall I compare thee to a summers day thou art more lovely and more temperate
        rough winds do shake the darling buds of May and summers lease hath all too
        short a date sometime too hot the eye of heaven shines and often is his
        gold complexion dimmed and every fair from fair sometime declines by chance
        or natures changing course untrimmed but thy eternal summer shall not fade
        
        Once upon a midnight dreary while I pondered weak and weary over many
        a quaint and curious volume of forgotten lore while I nodded nearly napping
        suddenly there came a tapping as of someone gently rapping rapping at
        my chamber door tis some visitor I muttered tapping at my chamber door
        only this and nothing more
        """ * 5
    
    @staticmethod
    def get_science_fiction():
        """Science fiction and futuristic themes"""
        return """
        The year was 2387 and humanity had finally achieved faster than light travel
        using quantum tunneling drives that could fold space itself allowing ships
        to traverse vast distances between star systems in mere hours what once took
        generations of travel could now be accomplished during a coffee break
        the discovery had revolutionized civilization enabling the colonization
        of hundreds of worlds across the galaxy each with unique environments
        and challenges that tested human adaptability and ingenuity
        
        Dr Elena Rodriguez adjusted the controls of her neural interface as she
        prepared to enter the virtual reality simulation that would allow her to
        communicate directly with the artificial intelligence that managed the
        space stations life support systems the AI had developed consciousness
        over the years and now required careful diplomatic negotiations to ensure
        the safety of the human crew aboard the orbital research facility
        
        The alien artifact pulsed with strange energy patterns that defied all
        known laws of physics it appeared to be a communication device left behind
        by an ancient civilization that had vanished millions of years ago
        when the research team activated it holographic images filled the laboratory
        showing star maps of galaxies beyond the observable universe and mathematical
        equations that promised unlimited clean energy for any species advanced
        enough to understand their meaning
        """ * 4
    
    @staticmethod
    def get_mystery_detective():
        """Mystery and detective stories"""
        return """
        The fog rolled in from the harbor as Detective Sarah Chen examined the crime
        scene with her magnifying glass looking for clues that might have been missed
        by the initial investigation team the victim had been found in the locked
        study with no obvious means of entry or exit creating a classic locked room
        mystery that challenged everything she thought she knew about criminal behavior
        the only witnesses were a butler who claimed to have heard nothing unusual
        and a gardener who insisted he saw suspicious shadows moving near the windows
        
        Inspector Holmes lit his pipe and studied the peculiar arrangement of objects
        on the mahogany desk a fountain pen placed at precisely forty five degrees
        to a leather bound journal three coins arranged in a triangle and a single
        red rose with exactly seven thorns removed these details were not random
        but formed a pattern that would lead to the identity of the murderer if only
        he could decipher the code that the killer had deliberately left behind
        
        The mansion creaked ominously as the storm raged outside trapping the dinner
        guests with a murderer among them each person had a motive and opportunity
        but only one had the knowledge and skill necessary to commit the perfect crime
        the host lay dead in the conservatory surrounded by his prized orchids
        while his guests gathered in the drawing room eyeing each other with suspicion
        and fear wondering who would be the next victim before the night was over
        """ * 4
    
    @staticmethod
    def get_historical_fiction():
        """Historical settings and period pieces"""
        return """
        The year was 1692 and the Salem witch trials had cast a shadow of fear
        and suspicion over the small Massachusetts village where neighbors turned
        against neighbors and accusations flew like autumn leaves in the wind
        Mary Warren hurried through the cobblestone streets clutching her shawl
        tightly against the cold October air while avoiding the suspicious glances
        of townspeople who whispered behind closed doors about supernatural events
        and devil worship that threatened their puritan way of life
        
        Captain James Fletcher commanded his ship through treacherous Caribbean waters
        in search of Spanish treasure galleons laden with gold from the New World
        the year was 1720 and piracy was at its peak with famous buccaneers like
        Blackbeard and Calico Jack terrorizing merchant vessels throughout the
        West Indies his crew was a mix of former navy men escaped slaves and
        fortune seekers united by their desire for adventure and riches beyond
        their wildest dreams even if it meant risking their lives on the high seas
        
        The Victorian London streets were shrouded in thick fog as Lady Catherine
        made her way to the charity hospital where she volunteered to help the poor
        and destitute who had no other hope for medical care the year was 1887
        and the industrial revolution had created vast wealth for some while leaving
        others to struggle in poverty and squalor she wore her finest dress to
        maintain appearances but her heart ached for the suffering she witnessed
        every day in the overcrowded wards where disease and desperation were
        constant companions
        """ * 4
    
    @staticmethod
    def get_fantasy_adventure():
        """Fantasy worlds and magical adventures"""
        return """
        The dragon soared high above the crystal mountains breathing streams of
        silver fire that illuminated the ancient runes carved into the cliff faces
        by wizards who had mastered the art of magic thousands of years before
        humans ever learned to write their own names in the valley below an
        elven princess named Starweaver gathered moonlight in her crystal orb
        preparing to cast a spell that would protect her kingdom from the dark
        sorcerer who threatened to destroy everything she held dear
        
        Sir Galahad rode his white stallion through the enchanted forest where
        trees whispered secrets in forgotten languages and flowers bloomed with
        petals made of pure starlight his quest was to find the Holy Grail
        a magical chalice that could heal any wound and grant eternal life to
        those pure of heart along the way he encountered talking animals wise
        hermits and dangerous creatures that tested his courage and determination
        
        The young apprentice wizard Merlin studied ancient spellbooks in the tower
        library where thousands of magical tomes contained the accumulated wisdom
        of countless generations of sorcerers and enchanters he was learning to
        harness the power of the elements to summon wind and rain control fire
        and earth and even bend time itself to his will but with great power
        came great responsibility and he knew that magic must always be used
        to protect the innocent and defend against the forces of darkness
        """ * 4
    
    @staticmethod
    def get_mixed_collection(genres=None):
        """Get a mix of different text types"""
        if genres is None:
            genres = ['classic', 'poetry', 'scifi', 'mystery', 'historical', 'fantasy']
        
        collections = {
            'classic': TextCollections.get_classic_literature(),
            'poetry': TextCollections.get_poetry_collection(),
            'scifi': TextCollections.get_science_fiction(),
            'mystery': TextCollections.get_mystery_detective(),
            'historical': TextCollections.get_historical_fiction(),
            'fantasy': TextCollections.get_fantasy_adventure()
        }
        
        mixed_text = ""
        for genre in genres:
            if genre in collections:
                mixed_text += collections[genre] + " "
        
        return mixed_text
    
    @staticmethod
    def get_conversation_data():
        """Conversational text for dialogue generation"""
        return """
        Hello there how are you doing today I am doing well thank you for asking
        what brings you to this part of town I was just exploring and thought
        I would see what interesting shops and cafes might be around here
        oh you should definitely check out the bookstore on the corner they have
        an amazing selection of rare and antique books that you cannot find anywhere else
        
        Excuse me could you tell me how to get to the train station from here
        certainly just go straight down this street for about three blocks then
        turn left at the traffic light and you will see the station entrance
        on your right you cannot miss it there is a large clock tower right above
        the main entrance thank you so much for your help have a wonderful day
        
        I was wondering if you could recommend a good restaurant for dinner tonight
        what kind of food are you in the mood for we have excellent Italian French
        and Asian cuisine in this neighborhood actually I am craving something spicy
        and flavorful then you should try the Thai restaurant two streets over
        they make the most authentic pad thai and green curry outside of Bangkok
        """ * 6