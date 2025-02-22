-- Индексы на таблицу Lecturers

-- Используем хеш индекс потому что у нас нет запросов на поиск
-- идентификаторов лекторов из диапазона, поиска наименьшего или
-- наибольшего идентификатора и так далее (в принципе отсутствует
-- порядок на суррогатных идентификаторах). А для поиска конкретного
-- значения эффективнее использовать хеши.
-- ДЗ-5.3.4. Информация о студентах с заданной оценкой по дисциплине,
-- которую у них вёл лектор, заданный ФИО.
create unique index IdxLecturerId on Lecturers using hash (LecturerId);

-- Используем хеш индекс потому что у нас нет запросов на поиск
-- имен лекторов из диапазона, поиска наименьшего или наибольшего
-- имен и так далее. А для поиска конкретного значения эффективнее
-- использовать хеши.
-- ДЗ-5.6.1. Идентификаторы студентов по преподавателю, имеющих хотя
-- бы одну оценку у преподавателя.
-- ДЗ-5.6.2. Идентификаторы студентов по преподавателю, не имеющих ни
-- одной оценки у преподавателя.
-- ДЗ-5.6.3. Идентификаторы студентов по преподавателю, имеющих
-- оценки по всем дисциплинам преподавателя.
create index IdxLecturerName on Lecturers using hash (LecturerName);
