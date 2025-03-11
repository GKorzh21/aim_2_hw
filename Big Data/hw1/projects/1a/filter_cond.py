def filter_cond(line_dict):
    """Функция фильтрации
    Принимает словарь с именами полей и их значениями в качестве аргумента
    Возвращает True, если условия выполнены
    """
    if1 = line_dict.get("if1")

    if if1 is None:
        return False
    try:
        int(if1)
    except ValueError:
        return False
    return 20 < int(if1) < 40
