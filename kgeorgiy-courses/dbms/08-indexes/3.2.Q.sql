-- Количество различных лекторов, читающих курсы у заданной
-- идентификатором группы.
select
    count(distinct LecturerId) as LecturersCount
from
    Plan
where
    GroupId = 1;
