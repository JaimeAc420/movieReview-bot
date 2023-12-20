import mods.module as md

def handle_response(message):
    if message == '!help':
        return 'Solo escribe tu opinon sobre el cine Colombiano.\n Cualquier opionion neutral no sera correctamente procesada'
    elif message == '!about':
        return 'Mi nombre es Jaime y este es mi bot'
    else:
        sentiment = md.predict_opinion(message)
        if sentiment == 'POSITIVE':
            return 'Me alegra mucho que disfrutes el cine Colombiano de esa manera!'
        elif sentiment == 'NEGATIVE':
            return 'Lo siento mucho! no tenia idea que no disfrutabas del cine Colombiano.. ¿y si le das otra oportunidad?'
    
