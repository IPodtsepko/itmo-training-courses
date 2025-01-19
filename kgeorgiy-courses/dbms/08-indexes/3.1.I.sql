-- Упорядоченный индекс для поиска строки по префиксу. Хеш индекс не
-- позволил бы быстрее чем с помощью полного перебора искать курс по
-- префиксу названия.
-- CourseId - небольшой идентификатор, но часто нужен именно он,
-- поэтому дополнительно делаем индекс покрывающим.
create index IdsOrderedCourseNameCourseId on Courses using btree (CourseName, CourseId);
