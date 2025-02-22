-- Индексы на таблицу Marks

-- Покрывающий индекс, который ускоряет join таблиц Courses, Marks
-- и Students, когда не требуется получать Mark.
-- ДЗ-5.4.1. Информация о студентах не имеющих оценки по дисциплине,
-- среди всех студентов.
-- ДЗ-5.4.2. Информация о студентах не имеющих оценки по дисциплине,
-- среди студентов, у которых есть эта дисциплина.
-- ДЗ-5.6.1. Идентификаторы студентов по преподавателю, имеющих хотя
-- бы одну оценку у преподавателя.
create index IdxCourseIdStudentId on Marks using btree (CourseId, StudentId);

-- Покрывающий индекс, который ускоряет поиск дисциплины, по которой
-- студент имеет хотя бы одну оценку.
-- ДЗ-5.7.1. Группы и дисциплины, такие что все студенты группы имеют
-- оценку по этой дисциплине (идентификаторы).
-- ДЗ-5.7.2. Группы и дисциплины, такие что все студенты группы имеют
-- оценку по этой дисциплине (названия).
create index IdxStudentIdCourseId on Marks using btree (StudentId, CourseId);

-- Покрывающий индекс, который ускоряет поиск оценки для студента,
-- когда не важно по какому курсу эта оценка была выставлена.
-- ДЗ-5.8.1. Суммарный балл одного студента.
-- ДЗ-5.8.2. Суммарный балл каждого студента.
-- ДЗ-5.9.1. Средний балл одного студента.
create index IdxStudentIdMark on Marks using btree (StudentId, Mark);
