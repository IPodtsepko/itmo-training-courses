BEGIN;

INSERT INTO
  Lecturers (LecturerId, LecturerName)
VALUES
  (1, 'Корнеев Г. А.'),
  (2, 'Станкевич А. С.'),
  (3, 'Кохась К. П.'),
  (4, 'Ведерников Н. В.');

INSERT INTO
  Courses (CourseId, CourseName)
VALUES
  (1, 'Математический анализ'),
  (2, 'Дискретная математика'),
  (3, 'Введение в программирование');

INSERT INTO
  Groups (GroupId, GroupName)
VALUES
  (1, 'M31341'),
  (2, 'M41351');

INSERT INTO
  Students (StudentId, StudentName, GroupId)
VALUES
  (1, 'Иванов И. И.', 1),
  (2, 'Петров П. П.', 1),
  (3, 'Сидоров С. С.', 2);

INSERT INTO
  Marks (StudentId, CourseId, Mark)
VALUES
  (1, 1, 1),
  (1, 2, 2),
  (1, 3, 3),
  (2, 1, 4),
  (2, 2, 5),
  (2, 3, 1),
  (3, 1, 2),
  (3, 3, 4);

INSERT INTO
  Plan (CourseId, GroupId, LecturerId)
VALUES
  (1, 1, 3),
  (1, 2, 4),
  (2, 1, 2),
  (3, 1, 1),
  (3, 2, 4);

END;
