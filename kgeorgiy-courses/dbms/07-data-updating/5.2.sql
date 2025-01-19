create view AllMarks as
select
    StudentId,
    (
        select
            count(*)
        from
            Marks M
        where
            S.StudentId = M.StudentId
    ) + (
        select
            count(*)
        from
            NewMarks M
        where
            S.StudentId = M.StudentId
    ) as Marks
from
    Students S;
